#!/usr/bin/env python3
"""Audit exact SFT token lengths for every model and prompt contract."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from prompt_contract import apply_native_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="key=/absolute/model/path")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize_ids(value: Any) -> list[int]:
    if hasattr(value, "get"):
        value = value.get("input_ids", value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def encode_plain(tokenizer: Any, prompt: str, answer: str) -> tuple[int, int, int]:
    prompt_ids = normalize_ids(tokenizer(prompt, add_special_tokens=False))
    answer_ids = normalize_ids(
        tokenizer(answer.rstrip() + (tokenizer.eos_token or ""), add_special_tokens=False)
    )
    return len(prompt_ids), len(answer_ids), len(prompt_ids) + len(answer_ids)


def encode_native(tokenizer: Any, prompt: str, answer: str) -> tuple[int, int, int]:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer has no native chat_template")
    user = [{"role": "user", "content": prompt}]
    prefix = normalize_ids(
        apply_native_template(tokenizer, user, tokenize=True, add_generation_prompt=True)
    )
    full = normalize_ids(
        apply_native_template(
            tokenizer,
            user + [{"role": "assistant", "content": answer.strip()}],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full[: len(prefix)] != prefix:
        raise ValueError("native template prefix mismatch")
    return len(prefix), len(full) - len(prefix), len(full)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(values: list[int]) -> dict[str, Any]:
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "p999": percentile(values, 0.999),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.train_file.open(encoding="utf-8")
        if line.strip()
    ]
    if not rows:
        raise ValueError("empty train file")

    from transformers import AutoTokenizer

    result: dict[str, Any] = {
        "train_file": str(args.train_file),
        "rows": len(rows),
        "response_sources": dict(Counter(str(row["response_source"]) for row in rows)),
        "models": {},
    }
    for spec in args.model:
        key, model_path = spec.split("=", 1)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model_result: dict[str, Any] = {}
        for mode in ("plain", "native"):
            prompt_lengths: list[int] = []
            answer_lengths: list[int] = []
            total_lengths: list[int] = []
            longest: list[dict[str, Any]] = []
            encoder = encode_plain if mode == "plain" else encode_native
            for index, row in enumerate(rows, 1):
                prompt_length, answer_length, total_length = encoder(
                    tokenizer, str(row["prompt"]), str(row["answer"])
                )
                prompt_lengths.append(prompt_length)
                answer_lengths.append(answer_length)
                total_lengths.append(total_length)
                longest.append(
                    {
                        "id": str(row["id"]),
                        "response_source": str(row["response_source"]),
                        "prompt_tokens": prompt_length,
                        "answer_tokens": answer_length,
                        "total_tokens": total_length,
                    }
                )
                if index % 5000 == 0:
                    print(f"[tokens] {key}/{mode} {index}/{len(rows)}", flush=True)
            longest.sort(key=lambda item: item["total_tokens"], reverse=True)
            model_result[mode] = {
                "prompt": summarize(prompt_lengths),
                "answer": summarize(answer_lengths),
                "total": summarize(total_lengths),
                "rows_over": {
                    str(limit): sum(length > limit for length in total_lengths)
                    for limit in (2048, 3072, 4096, 6144, 8192)
                },
                "longest_rows": longest[:20],
            }
        result["models"][key] = model_result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
