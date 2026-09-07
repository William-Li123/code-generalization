#!/usr/bin/env python3
"""Generate and execute the held-out verified SFT validation split."""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from prompt_contract import apply_native_template


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--mbpp-file", type=Path, required=True)
    parser.add_argument("--mmlu-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-mode", choices=["plain", "native"], required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--grader-workers", type=int, default=32)
    parser.add_argument("--choice-batch-size", type=int, default=48)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def render(tokenizer: Any, prompt: str, mode: str) -> str:
    if mode == "plain":
        return prompt
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("native mode requested but tokenizer has no chat_template")
    return apply_native_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    device = next(model.parameters()).device

    private_rows = read_jsonl(args.data_file)
    prompts = [render(tokenizer, row["prompt"], args.prompt_mode) for row in private_rows]
    replies: list[str] = []
    generated_tokens: list[int] = []
    truncated: list[bool] = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(prompts), args.batch_size):
            chunk = prompts[start : start + args.batch_size]
            # render(native) already includes the tokenizer's native control
            # tokens; plain mode is deliberately raw. This matches the formal
            # evaluator's add_special_tokens=False contract for both modes.
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            prompt_width = encoded["input_ids"].shape[1]
            outputs = model.generate(
                **encoded,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for output in outputs:
                continuation = output[prompt_width:]
                count = int(continuation.numel())
                generated_tokens.append(count)
                truncated.append(count >= args.max_new_tokens)
                replies.append(tokenizer.decode(continuation, skip_special_tokens=True))
            print(
                f"[internal] private_code generated {min(start + len(chunk), len(prompts))}/{len(prompts)}",
                flush=True,
            )

    sys.path.insert(0, str(REPO_ROOT / "stages" / "stage05_legacy_taco_dapo_ablation"))
    from stage3_code_reward import compute_score

    def grade(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
        row, reply = item
        result = compute_score(
            data_source=f"sft_diagnise_{row['dataset']}",
            solution_str=reply,
            ground_truth=row["ground_truth"],
            exec_timeout=20.0,
            per_test_timeout=4.0,
        )
        return result

    with ThreadPoolExecutor(max_workers=args.grader_workers) as executor:
        scores = list(executor.map(grade, zip(private_rows, replies)))
    print(f"[internal] private_code graded {len(scores)}/{len(private_rows)}", flush=True)

    private_records: list[dict[str, Any]] = []
    statuses = Counter()
    fractions: list[float] = []
    private_passed = 0
    for row, reply, result, token_count, was_truncated in zip(
        private_rows, replies, scores, generated_tokens, truncated, strict=True
    ):
        is_passed = float(result.get("passed", 0.0)) == 1.0
        private_passed += int(is_passed)
        status = str(result.get("status") or result.get("status_name") or result.get("status_code"))
        statuses[status] += 1
        fraction = float(
            result.get(
                "pass_frac",
                result.get("test_pass_fraction", result.get("pass_ratio", result.get("score", int(is_passed)))),
            )
        )
        fractions.append(fraction)
        private_records.append(
            {
                "id": row["id"],
                "component": "private_code",
                "dataset": row["dataset"],
                "output_contract": row["output_contract"],
                "source_bucket": row["source_bucket"],
                "passed": is_passed,
                "test_pass_fraction": fraction,
                "status": status,
                "generated_tokens": token_count,
                "truncated": was_truncated,
                "reply": reply,
                "score_detail": result,
            }
        )

    sys.path.insert(0, str(REPO_ROOT / "evaluation"))
    from legacy_suite import (
        extract_code,
        generate_batch,
        mbpp_entry_point,
        render_mbpp_prompt,
        render_prompt,
        rewrite_single_function_name,
        run_python,
        score_choices_batch,
    )

    formal_mode = "chat" if args.prompt_mode == "native" else "plain"
    formal_args = SimpleNamespace(prompt_mode=formal_mode)
    mbpp_rows = read_jsonl(args.mbpp_file)
    print(f"[internal] mbpp_code generation start n={len(mbpp_rows)}", flush=True)
    mbpp_prompts = [render_mbpp_prompt(tokenizer, row, formal_args) for row in mbpp_rows]
    mbpp_replies = generate_batch(
        model,
        tokenizer,
        mbpp_prompts,
        args.max_new_tokens,
        args.batch_size,
        args.temperature,
        args.top_p,
    )

    def grade_mbpp(item: tuple[dict[str, Any], str]) -> tuple[bool, str]:
        row, reply = item
        try:
            code = extract_code(reply)
            code = rewrite_single_function_name(code, mbpp_entry_point(row))
            imports = "\n".join(row.get("test_imports") or [])
            tests = "\n".join(row.get("test_list") or [])
            ast.parse(code)
            return run_python(imports + "\n\n" + code + "\n\n" + tests + "\n", timeout_seconds=12)
        except Exception as exc:
            return False, f"{type(exc).__name__}:{exc}"

    with ThreadPoolExecutor(max_workers=min(args.grader_workers, 16)) as executor:
        mbpp_scores = list(executor.map(grade_mbpp, zip(mbpp_rows, mbpp_replies)))
    print(f"[internal] mbpp_code graded {len(mbpp_scores)}/{len(mbpp_rows)}", flush=True)
    mbpp_records: list[dict[str, Any]] = []
    mbpp_tokens = [len(tokenizer(reply, add_special_tokens=False)["input_ids"]) for reply in mbpp_replies]
    for row, reply, (is_passed, detail), token_count in zip(
        mbpp_rows, mbpp_replies, mbpp_scores, mbpp_tokens, strict=True
    ):
        mbpp_records.append(
            {
                "id": row["id"],
                "component": "mbpp_code",
                "dataset": "mbpp_aux_validation",
                "passed": bool(is_passed),
                "test_pass_fraction": float(is_passed),
                "status": "passed" if is_passed else "failed",
                "generated_tokens": token_count,
                "truncated": token_count >= args.max_new_tokens,
                "reply": reply,
                "score_detail": detail,
            }
        )

    mmlu_rows = read_jsonl(args.mmlu_file)
    print(f"[internal] mmlu_math scoring start n={len(mmlu_rows)}", flush=True)
    labels = ["A", "B", "C", "D"]

    def mmlu_prompt(row: dict[str, Any]) -> str:
        options = "\n".join(
            f"{labels[index]}. {choice}" for index, choice in enumerate(row["choices"])
        )
        subject = str(row["subject"]).replace("_", " ")
        text = (
            f"Answer the following multiple-choice mathematics question ({subject}).\n\n"
            f"Question: {row['question']}\n{options}\n\n"
            "Answer with only A, B, C, or D.\nAnswer:"
        )
        return render_prompt(tokenizer, text, formal_mode)

    mmlu_prompts = [mmlu_prompt(row) for row in mmlu_rows]
    mmlu_predictions = score_choices_batch(
        model,
        tokenizer,
        mmlu_prompts,
        [labels for _ in mmlu_rows],
        args.choice_batch_size,
    )
    print(f"[internal] mmlu_math scored {len(mmlu_predictions)}/{len(mmlu_rows)}", flush=True)
    mmlu_records: list[dict[str, Any]] = []
    for row, (prediction, values) in zip(mmlu_rows, mmlu_predictions, strict=True):
        is_passed = int(prediction) == int(row["answer"])
        mmlu_records.append(
            {
                "id": row["id"],
                "component": "mmlu_math",
                "dataset": "mmlu_math_validation",
                "subject": row["subject"],
                "passed": is_passed,
                "test_pass_fraction": float(is_passed),
                "status": "correct" if is_passed else "incorrect",
                "prediction": labels[int(prediction)],
                "target": labels[int(row["answer"])],
                "choice_scores": [float(value) for value in values],
                "truncated": False,
            }
        )

    records = private_records + mbpp_records + mmlu_records
    for filename, values in (
        ("private_code_records.jsonl", private_records),
        ("mbpp_code_records.jsonl", mbpp_records),
        ("mmlu_math_records.jsonl", mmlu_records),
        ("records.jsonl", records),
    ):
        with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
            for record in values:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def component_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(int(row["passed"]) for row in values)
        return {"n": len(values), "correct": correct, "accuracy": correct / max(1, len(values))}

    mmlu_by_subject: dict[str, dict[str, Any]] = {}
    for subject in sorted({str(row["subject"]) for row in mmlu_records}):
        subject_rows = [row for row in mmlu_records if str(row["subject"]) == subject]
        mmlu_by_subject[subject] = component_summary(subject_rows)
    mmlu_micro = component_summary(mmlu_records)
    mmlu_component = {
        **mmlu_micro,
        "accuracy": sum(item["accuracy"] for item in mmlu_by_subject.values())
        / max(1, len(mmlu_by_subject)),
        "micro_accuracy": mmlu_micro["accuracy"],
        "aggregation": "macro over five mathematics subjects",
        "by_subject": mmlu_by_subject,
    }
    components = {
        "private_code": component_summary(private_records),
        "mbpp_code": component_summary(mbpp_records),
        "mmlu_math": mmlu_component,
    }
    mixed_macro_accuracy = sum(item["accuracy"] for item in components.values()) / len(components)
    code_macro_accuracy = (
        components["private_code"]["accuracy"] + components["mbpp_code"]["accuracy"]
    ) / 2
    all_generated_tokens = generated_tokens + mbpp_tokens
    all_truncated = truncated + [row["truncated"] for row in mbpp_records]
    passed = sum(int(row["passed"]) for row in records)
    all_fractions = [float(row["test_pass_fraction"]) for row in records]
    summary = {
        "model_key": args.model_key,
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "data_file": str(args.data_file),
        "mbpp_file": str(args.mbpp_file),
        "mmlu_file": str(args.mmlu_file),
        "prompt_mode": args.prompt_mode,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "n": len(records),
        "passed": passed,
        "pass_rate": passed / max(1, len(records)),
        "mixed_macro_accuracy": mixed_macro_accuracy,
        "code_macro_accuracy": code_macro_accuracy,
        "components": components,
        "mean_test_pass_fraction": sum(all_fractions) / max(1, len(all_fractions)),
        "truncated": sum(all_truncated),
        "truncation_rate": sum(all_truncated) / max(1, len(all_truncated)),
        "mean_generated_tokens": sum(all_generated_tokens) / max(1, len(all_generated_tokens)),
        "status_counts": dict(statuses),
        "wall_seconds": round(time.time() - started, 2),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
