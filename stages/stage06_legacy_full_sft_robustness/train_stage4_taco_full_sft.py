#!/usr/bin/env python3
"""Train one Stage 4 full-parameter SFT checkpoint with assistant-only loss."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset


@dataclass
class RunConfig:
    model_name: str
    model_path: str
    arm: str
    train_file: str
    checkpoint_dir: str
    log_dir: str
    seed: int
    max_seq_len: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    max_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    fsdp: str
    fsdp_layer_cls: str
    gradient_checkpointing: bool
    save_only_model: bool
    save_dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--fsdp", default="full_shard auto_wrap")
    parser.add_argument("--fsdp-layer-cls", default="")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-only-model", action="store_true", default=True)
    parser.add_argument("--save-dtype", choices=["bf16", "fp32"], default="bf16")
    return parser.parse_args()


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_rank0() -> bool:
    return rank() == 0


def infer_fsdp_layer_cls(model_name: str, model_path: str) -> str:
    config_path = Path(model_path) / "config.json"
    model_type = ""
    if config_path.exists():
        try:
            model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type", "")
        except Exception:
            model_type = ""
    if model_type == "qwen3" or "Qwen3" in model_name:
        return "Qwen3DecoderLayer"
    return "Qwen2DecoderLayer"


def encode_example(tokenizer, row: dict[str, Any], max_seq_len: int) -> tuple[list[int], list[int], bool]:
    prompt = row.get("prompt") or (row.get("messages") or [{}])[0].get("content", "")
    answer = row.get("answer") or (row.get("messages") or [{}, {}])[1].get("content", "")
    prefix = prompt
    suffix = answer.rstrip() + (tokenizer.eos_token or "")
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    truncated = False
    if len(prefix_ids) + len(answer_ids) > max_seq_len:
        truncated = True
        answer_budget = max(256, min(len(answer_ids), max_seq_len // 2))
        prompt_budget = max_seq_len - answer_budget
        if len(prefix_ids) > prompt_budget:
            prefix_ids = prefix_ids[:prompt_budget]
        answer_budget = max_seq_len - len(prefix_ids)
        answer_ids = answer_ids[:answer_budget]
    input_ids = prefix_ids + answer_ids
    labels = [-100] * len(prefix_ids) + answer_ids[:]
    return input_ids, labels, truncated


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def build_dataset(tokenizer, train_file: Path, max_seq_len: int, log_dir: Path) -> Dataset:
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    truncated_count = 0
    skipped_empty_answer = 0
    token_lengths: list[int] = []
    with train_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            answer = row.get("answer") or ""
            if not answer.strip():
                skipped_empty_answer += 1
                continue
            ids, labs, truncated = encode_example(tokenizer, row, max_seq_len)
            input_ids.append(ids)
            labels.append(labs)
            token_lengths.append(len(ids))
            truncated_count += int(truncated)
    stats = {
        "train_file": str(train_file),
        "examples": len(input_ids),
        "skipped_empty_answer": skipped_empty_answer,
        "truncated_count": truncated_count,
        "max_seq_len": max_seq_len,
        "mean_tokens": sum(token_lengths) / max(len(token_lengths), 1),
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "p50_tokens": percentile(token_lengths, 0.50),
        "p95_tokens": percentile(token_lengths, 0.95),
        "p99_tokens": percentile(token_lengths, 0.99),
    }
    if is_rank0():
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels})


class DataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        length = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full((len(rows), length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), length), -100, dtype=torch.long)
        attention = torch.zeros((len(rows), length), dtype=torch.long)
        for index, row in enumerate(rows):
            count = len(row["input_ids"])
            input_ids[index, :count] = torch.tensor(row["input_ids"], dtype=torch.long)
            labels[index, :count] = torch.tensor(row["labels"], dtype=torch.long)
            attention[index, :count] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def rolling_loss_summary(log_path: Path) -> dict[str, Any]:
    losses: list[float] = []
    if log_path.exists():
        with log_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if "loss" in row:
                    losses.append(float(row["loss"]))
    if not losses:
        return {"loss_count": 0}
    last = losses[-20:]
    mean = sum(last) / len(last)
    return {
        "loss_count": len(losses),
        "first_loss": losses[0],
        "final_logged_loss": losses[-1],
        "min_loss": min(losses),
        "last20_mean_loss": mean,
        "last20_std_loss": math.sqrt(sum((x - mean) ** 2 for x in last) / len(last)),
    }


def save_full_model_with_dtype(trainer, checkpoint_dir: Path, save_dtype: str) -> None:
    state_dict = trainer.accelerator.get_state_dict(trainer.model)
    if save_dtype == "bf16":
        state_dict = {
            key: value.to(dtype=torch.bfloat16) if torch.is_tensor(value) and torch.is_floating_point(value) else value
            for key, value in state_dict.items()
        }
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped_model.save_pretrained(
        str(checkpoint_dir),
        is_main_process=trainer.accelerator.is_main_process,
        save_function=trainer.accelerator.save,
        state_dict=state_dict,
        safe_serialization=True,
    )


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    if is_rank0():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    fsdp_layer_cls = args.fsdp_layer_cls or infer_fsdp_layer_cls(args.model_name, args.model_path)
    config = RunConfig(
        model_name=args.model_name,
        model_path=args.model_path,
        arm=args.arm,
        train_file=args.train_file,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        fsdp=args.fsdp,
        fsdp_layer_cls=fsdp_layer_cls,
        gradient_checkpointing=args.gradient_checkpointing,
        save_only_model=args.save_only_model,
        save_dtype=args.save_dtype,
    )
    if is_rank0():
        (log_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset(tokenizer, Path(args.train_file), args.max_seq_len, log_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    # With FSDP full_shard, HuggingFace recommends FSDP activation checkpointing
    # instead of TrainingArguments.gradient_checkpointing to avoid redundant
    # all-gathers in the backward pass.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log_path = log_dir / "train_loss.jsonl"
    memory_log_path = log_dir / f"memory_rank{rank()}.jsonl"

    class JsonlLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and is_rank0():
                record = {"step": int(state.global_step), "epoch": float(state.epoch or 0.0), **logs}
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available() and int(state.global_step) % max(1, config.logging_steps) == 0:
                record = {
                    "step": int(state.global_step),
                    "rank": rank(),
                    "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
                    "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
                    "max_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                }
                with memory_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")

    training_kwargs: dict[str, Any] = {}
    if args.max_steps > 0:
        training_kwargs["max_steps"] = args.max_steps
    else:
        training_kwargs["num_train_epochs"] = args.num_train_epochs
    training_args = TrainingArguments(
        output_dir=str(log_dir / "_trainer"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="no",
        save_safetensors=True,
        save_only_model=args.save_only_model,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        fsdp=args.fsdp,
        fsdp_config={
            "transformer_layer_cls_to_wrap": fsdp_layer_cls,
            "use_orig_params": False,
            "limit_all_gathers": True,
            "activation_checkpointing": args.gradient_checkpointing,
        },
        **training_kwargs,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollator(tokenizer.pad_token_id),
        callbacks=[JsonlLogCallback()],
    )
    started = time.time()
    result = trainer.train()
    wall_seconds = round(time.time() - started, 1)
    save_full_model_with_dtype(trainer, checkpoint_dir, args.save_dtype)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(checkpoint_dir))
        completed_steps = int(trainer.state.global_step)
        effective_batch = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * max(1, int(os.environ.get("WORLD_SIZE", "1")))
        )
        summary = {
            **asdict(config),
            "examples": len(dataset),
            "effective_global_batch": effective_batch,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_ratio": trainable / max(total, 1),
            "trainer_training_loss": float(result.training_loss),
            "completed_steps": completed_steps,
            "wall_seconds": wall_seconds,
            "sec_per_step": wall_seconds / max(completed_steps, 1),
            **rolling_loss_summary(log_path),
        }
        (checkpoint_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (log_dir / "training_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
