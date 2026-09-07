#!/usr/bin/env python3
"""Audit native chat templates and token lengths without repository-local data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must use KEY=PATH_OR_HF_ID syntax")
    key, reference = value.split("=", 1)
    if not key.strip() or not reference.strip():
        raise argparse.ArgumentTypeError("--model requires non-empty KEY and PATH_OR_HF_ID")
    return key.strip(), reference.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="External training rows in Parquet or JSONL format.",
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        metavar="KEY=PATH_OR_HF_ID",
        help="Repeat for each model to audit.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Transformers to fetch model/tokenizer files instead of requiring a local cache.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Opt in to executing model-repository Python code.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"--source does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=["problem_id", "prompt", "response"])
    elif path.suffix.lower() == ".jsonl":
        frame = pd.read_json(path, lines=True)
        missing = sorted({"problem_id", "prompt", "response"} - set(frame.columns))
        if missing:
            raise ValueError(f"JSONL source is missing columns: {missing}")
        frame = frame[["problem_id", "prompt", "response"]]
    else:
        raise ValueError("--source must have a .parquet or .jsonl suffix")
    if frame.empty:
        raise ValueError("--source contains no rows")
    if frame.isna().any().any():
        raise ValueError("problem_id, prompt, and response must not contain null values")
    return frame.to_dict(orient="records")


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def audit_model(
    key: str,
    model_reference: str,
    rows: list[dict[str, str]],
    max_length: int,
    batch_size: int,
    allow_downloads: bool,
    trust_remote_code: bool,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_reference,
        trust_remote_code=trust_remote_code,
        local_files_only=not allow_downloads,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{key} has no native chat_template")
    lengths: list[int] = []
    overlength: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for row in chunk
        ]
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=False,
            return_length=True,
        )
        chunk_lengths = [int(value) for value in encoded["length"]]
        lengths.extend(chunk_lengths)
        for row, length in zip(chunk, chunk_lengths, strict=True):
            if length > max_length and len(overlength) < 100:
                overlength.append({"problem_id": row["problem_id"], "tokens": length})
    return {
        "model_key": key,
        "model_reference": model_reference,
        "rows": len(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "p95_tokens": percentile(lengths, 0.95),
        "p99_tokens": percentile(lengths, 0.99),
        "max_tokens": max(lengths),
        "overlength_rows": sum(length > max_length for length in lengths),
        "overlength_examples": overlength,
        "native_template": True,
        "enable_thinking": False,
    }


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    rows = read_rows(source)
    models = dict(args.model)
    if len(models) != len(args.model):
        raise ValueError("duplicate model keys were supplied")
    if args.max_sequence_length <= 0 or args.batch_size <= 0:
        raise ValueError("--max-sequence-length and --batch-size must be positive")

    results = [
        audit_model(
            key,
            reference,
            rows,
            args.max_sequence_length,
            args.batch_size,
            args.allow_downloads,
            args.trust_remote_code,
        )
        for key, reference in models.items()
    ]
    report = {
        "schema_version": 1,
        "source_name": source.name,
        "source_sha256": sha256(source),
        "source_rows": len(rows),
        "max_sequence_length": args.max_sequence_length,
        "passed": all(item["overlength_rows"] == 0 for item in results),
        "models": results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
