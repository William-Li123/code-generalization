#!/usr/bin/env python3
"""Recompute tokenizer lengths and finalize the verified Stage 05 split."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from prepare_stage3_dapo_full_data import allocate_validation, summarize, token_length


DATA_ROOT = Path(os.environ["CG_DATA_ROOT"]).expanduser() if os.environ.get("CG_DATA_ROOT") else None
MODEL_ROOT = Path(os.environ["CG_MODEL_ROOT"]).expanduser() if os.environ.get("CG_MODEL_ROOT") else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_ROOT / "stage3_dapo_full_verified" if DATA_ROOT else None,
        required=DATA_ROOT is None,
    )
    parser.add_argument(
        "--model-paths",
        type=Path,
        nargs="+",
        default=(
            [MODEL_ROOT / "Qwen3-8B-Base", MODEL_ROOT / "Qwen2.5-7B-Instruct"]
            if MODEL_ROOT
            else None
        ),
        required=MODEL_ROOT is None,
    )
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260609)
    return parser.parse_args()


def write_parquet(path: Path, records: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")


def main() -> None:
    args = parse_args()
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict] = []
    for name in ("train.parquet", "val.parquet"):
        records.extend(pq.read_table(args.data_dir / name).to_pylist())

    tokenizers = [
        AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        for path in args.model_paths
    ]
    accepted: list[dict] = []
    overlong: list[str] = []
    for record in records:
        lengths = [token_length(tokenizer, record["prompt"]) for tokenizer in tokenizers]
        record["extra_info"]["prompt_tokens_by_model"] = dict(
            zip(map(str, args.model_paths), lengths, strict=True)
        )
        record["extra_info"]["prompt_tokens_max"] = max(lengths)
        if max(lengths) > args.max_prompt_length:
            overlong.append(record["extra_info"]["problem_id"])
        else:
            accepted.append(record)

    random.Random(args.seed).shuffle(accepted)
    allocation = allocate_validation(accepted, args.val_count)
    used: Counter[str] = Counter()
    train: list[dict] = []
    val: list[dict] = []
    for record in accepted:
        key = record["extra_info"]["interface_kind"]
        if used[key] < allocation[key]:
            val.append(record)
            used[key] += 1
        else:
            train.append(record)
    for index, record in enumerate(train):
        record["extra_info"]["index"] = index
    for index, record in enumerate(val):
        record["extra_info"]["index"] = index

    write_parquet(args.data_dir / "train.parquet", train)
    write_parquet(args.data_dir / "val.parquet", val)
    with (args.data_dir / "validation100.jsonl").open("w", encoding="utf-8") as handle:
        for record in val:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest["accepted_rows_before_holdout"] = len(accepted)
    manifest["validation_allocation"] = allocation
    manifest["train"] = summarize(train)
    manifest["validation"] = summarize(val)
    manifest.setdefault("reject_counts", {})["prompt_overlong"] = len(overlong)
    manifest.setdefault("reject_examples", {})["prompt_overlong"] = overlong[:20]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=dict),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
