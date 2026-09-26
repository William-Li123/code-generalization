#!/usr/bin/env python3
"""Train one Stage 1 assistant-only LoRA arm with a native model template."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset


TARGET_PROFILES = {
    "qv": ["q_proj", "v_proj"],
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-mode", choices=["plain", "native"], required=True)
    parser.add_argument("--target-profile", choices=["auto", *sorted(TARGET_PROFILES)], default="auto")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-resume", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalize_ids(value: Any) -> list[int]:
    if hasattr(value, "get"):
        value = value.get("input_ids", value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_plain(tokenizer: Any, prompt: str, answer: str) -> tuple[list[int], list[int]]:
    prefix = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    suffix = list(
        tokenizer(answer.rstrip() + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    )
    return prefix + suffix, [-100] * len(prefix) + suffix


def encode_native(tokenizer: Any, prompt: str, answer: str) -> tuple[list[int], list[int]]:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("native mode requested but tokenizer has no chat_template")
    answer = answer.strip()
    if not answer:
        raise ValueError("native template received an empty assistant answer")
    user = [{"role": "user", "content": prompt}]
    prefix = normalize_ids(
        tokenizer.apply_chat_template(
            user,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    full = normalize_ids(
        tokenizer.apply_chat_template(
            user + [{"role": "assistant", "content": answer}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if full[: len(prefix)] == prefix:
        if len(full) == len(prefix):
            raise ValueError("native template produced an empty assistant suffix")
        return full, [-100] * len(prefix) + full[len(prefix) :]

    # Some BPE tokenizers re-tokenize the final generation-marker token once
    # the following newline/answer is present. In that case the rendered text
    # is still a valid prefix even though the token IDs are not an exact prefix.
    prefix_text = tokenizer.apply_chat_template(
        user,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not full_text.startswith(prefix_text):
        raise ValueError("native template text prefix mismatch")
    answer_start = full_text.find(answer, len(prefix_text))
    if answer_start < 0:
        raise ValueError("native template assistant answer boundary not found")

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offset_full = normalize_ids(encoded)
    if offset_full != full:
        raise ValueError("native template tokenization is inconsistent")
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("native template fallback requires tokenizer offset mapping")
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    label_start = next(
        (
            index
            for index, (start, end) in enumerate(offsets)
            if int(end) > answer_start and int(end) > int(start)
        ),
        len(full),
    )
    if label_start >= len(full):
        raise ValueError("native template produced an empty assistant suffix")
    return full, [-100] * label_start + full[label_start:]


def read_training_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path, columns=["problem_id", "prompt", "response"])
        return frame.to_dict(orient="records")
    if path.suffix in {".jsonl", ".json"}:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError(f"unsupported training file: {path}")


def build_dataset(tokenizer: Any, path: Path, prompt_mode: str, max_seq_len: int) -> tuple[Dataset, dict[str, Any]]:
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    lengths: list[int] = []
    supervised: list[int] = []
    for line_number, row in enumerate(read_training_rows(path), 1):
        answer = row.get("response", row.get("answer"))
        if not isinstance(row.get("prompt"), str) or not isinstance(answer, str):
            raise ValueError(f"row {line_number} is missing string prompt/response")
        encoded, row_labels = (
            encode_plain(tokenizer, row["prompt"], answer)
            if prompt_mode == "plain"
            else encode_native(tokenizer, row["prompt"], answer)
        )
        if len(encoded) > max_seq_len:
            row_id = row.get("problem_id", row.get("id", line_number))
            raise ValueError(f"row {row_id} has {len(encoded)} tokens > {max_seq_len}")
        count = sum(label != -100 for label in row_labels)
        if count <= 0:
            raise ValueError(f"row {line_number} has no supervised assistant token")
        input_ids.append(encoded)
        labels.append(row_labels)
        lengths.append(len(encoded))
        supervised.append(count)
    if not input_ids:
        raise ValueError("empty training dataset")
    ordered = sorted(lengths)
    stats = {
        "rows": len(input_ids),
        "max_seq_len": max_seq_len,
        "truncated_rows": 0,
        "mean_tokens": sum(lengths) / len(lengths),
        "p95_tokens": ordered[int(0.95 * (len(ordered) - 1))],
        "max_tokens": max(lengths),
        "mean_supervised_tokens": sum(supervised) / len(supervised),
        "assistant_only_loss": True,
    }
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels}), stats


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        length = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full((len(rows), length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
        for index, row in enumerate(rows):
            count = len(row["input_ids"])
            input_ids[index, :count] = torch.tensor(row["input_ids"], dtype=torch.long)
            labels[index, :count] = torch.tensor(row["labels"], dtype=torch.long)
            attention_mask[index, :count] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def resolve_target_modules(model: Any, profile: str) -> list[str]:
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    candidates = (
        [TARGET_PROFILES["attention"], ["c_attn", "c_proj"]]
        if profile == "auto"
        else [TARGET_PROFILES[profile]]
    )
    for modules in candidates:
        if all(module in available for module in modules):
            return modules
    raise ValueError(
        f"no compatible LoRA target profile; requested={profile}, "
        f"available projection-like modules={sorted(name for name in available if 'proj' in name or 'attn' in name)}"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset, dataset_stats = build_dataset(tokenizer, args.train_file, args.prompt_mode, args.max_seq_len)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    target_modules = resolve_target_modules(model, args.target_profile)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        ),
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())

    log_path = args.output_dir / "train_log.jsonl"

    class JsonlCallback(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if logs and state.is_world_process_zero:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": int(state.global_step), "epoch": state.epoch, **logs}) + "\n")

    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir / "trainer_checkpoints"),
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "logging_steps": args.logging_steps,
        "bf16": True,
        "tf32": True,
        "optim": "adamw_torch_fused",
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
        "ddp_find_unused_parameters": False,
        "save_strategy": "steps" if args.save_resume else "no",
        "save_steps": 0.5 if args.save_resume else 500,
        "save_total_limit": 1,
    }
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=Collator(tokenizer.pad_token_id),
        callbacks=[JsonlCallback()],
    )
    started = time.time()
    resume_checkpoint: str | None = None
    if args.resume:
        candidates = [
            path
            for path in (args.output_dir / "trainer_checkpoints").glob("checkpoint-*")
            if (path / "trainer_state.json").is_file() and (path / "optimizer.pt").is_file()
        ]
        if candidates:
            latest = max(candidates, key=lambda path: int(path.name.rsplit("-", 1)[-1]))
            resume_checkpoint = str(latest)
            print(f"[resume] using {resume_checkpoint}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    wall_seconds = round(time.time() - started, 1)
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_dir))
        summary = {
            "model_key": args.model_key,
            "model_path": str(args.model_path),
            "train_file": str(args.train_file),
            "train_file_sha256": sha256(args.train_file),
            "prompt_mode": args.prompt_mode,
            "target_profile": args.target_profile,
            "target_modules": target_modules,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "effective_batch_size": args.per_device_batch_size
            * args.gradient_accumulation_steps
            * int(os.environ.get("WORLD_SIZE", "1")),
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "completed_steps": int(trainer.state.global_step),
            "training_loss": float(result.training_loss),
            "wall_seconds": wall_seconds,
            "dataset": dataset_stats,
            "final_adapter": str(final_dir),
            "adapter_checkpoints": sorted(
                str(path)
                for path in (args.output_dir / "trainer_checkpoints").glob("checkpoint-*")
                if (path / "adapter_model.safetensors").is_file()
            ),
            "full_model_checkpoint_saved": False,
            "optimizer_resume_state_saved": True,
            "resumed_from_checkpoint": resume_checkpoint,
            "enable_thinking": False,
        }
        (args.output_dir / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "COMPLETED").write_text(
            json.dumps(
                {
                    "completed_steps": int(trainer.state.global_step),
                    "final_adapter": str(final_dir),
                    "train_file_sha256": summary["train_file_sha256"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
