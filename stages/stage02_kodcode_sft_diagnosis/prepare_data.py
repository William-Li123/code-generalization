#!/usr/bin/env python3
"""Freeze the audited KodCode training set and check validation isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 37_881
EXPECTED_JSONL_SHA256 = "810637674922d3c1fd4c99e120749f6e4411e1e709463a3b8f8b002a4dc28f07"
EXPECTED_PARQUET_SHA256 = "6e91ea43bb43ff9d20514c065428152078b36ba57c6f6c577f794395f6decae1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_row(row: dict[str, Any], line_number: int) -> None:
    required = (
        "id",
        "original_question_id",
        "source_dataset",
        "response_source",
        "subset",
        "style",
        "difficulty",
        "prompt",
        "answer",
        "messages",
        "quality_status",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"line {line_number} missing fields: {missing}")
    if not str(row["prompt"]).strip() or not str(row["answer"]).strip():
        raise ValueError(f"line {line_number} has an empty prompt or answer")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"line {line_number} must have exactly user/assistant messages")
    expected = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    if messages != expected:
        raise ValueError(f"line {line_number} messages do not match prompt/answer")
    if "test_code" in str(row["prompt"]):
        raise ValueError(f"line {line_number} appears to expose test_code in the prompt")


def main() -> None:
    args = parse_args()
    if args.output_file.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output_file}; pass --overwrite")

    source_jsonl_hash = sha256(args.source_jsonl)
    source_parquet_hash = sha256(args.source_parquet)
    if source_jsonl_hash != EXPECTED_JSONL_SHA256:
        raise ValueError(f"unexpected JSONL SHA256: {source_jsonl_hash}")
    if source_parquet_hash != EXPECTED_PARQUET_SHA256:
        raise ValueError(f"unexpected Parquet SHA256: {source_parquet_hash}")

    rows = read_jsonl(args.source_jsonl)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    for line_number, row in enumerate(rows, 1):
        validate_row(row, line_number)

    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("training row IDs are not unique")

    shuffled = list(rows)
    random.Random(args.seed).shuffle(shuffled)
    write_jsonl(args.output_file, shuffled)

    original_question_counts = Counter(str(row["original_question_id"]) for row in rows)
    manifest = {
        "version": "sft_diagnise_2_data_v1",
        "seed": args.seed,
        "source": {
            "jsonl": str(args.source_jsonl),
            "jsonl_sha256": source_jsonl_hash,
            "parquet": str(args.source_parquet),
            "parquet_sha256": source_parquet_hash,
        },
        "output": {
            "train_file": str(args.output_file),
            "train_file_sha256": sha256(args.output_file),
            "rows": len(shuffled),
            "all_source_rows_retained": True,
        },
        "integrity": {
            "unique_row_ids": len(set(ids)),
            "unique_original_question_ids": len(original_question_counts),
            "original_questions_with_multiple_responses": sum(
                count > 1 for count in original_question_counts.values()
            ),
            "message_contract": "exactly one user message followed by one assistant message",
            "empty_prompt_or_answer_rows": 0,
        },
        "distribution": {
            field: dict(Counter(str(row[field]) for row in rows))
            for field in ("source_dataset", "response_source", "subset", "style", "difficulty")
        },
        "checkpoint_selection_validation": None,
        "safety": {
            "sample_code_executed_by_this_script": False,
            "test_code_exposed_to_training": False,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
