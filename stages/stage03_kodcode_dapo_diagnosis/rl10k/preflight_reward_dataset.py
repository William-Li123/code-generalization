#!/usr/bin/env python3
"""CPU-only preflight of retained reference answers through the exact RL reward."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


STAGE_ROOT = Path(__file__).resolve().parents[1]


def env_path(name: str, *parts: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().joinpath(*parts) if value else None


def parse_args() -> argparse.Namespace:
    default_source = env_path("CG_DATA_ROOT", "kodcode4o_r1_clean38k", "train.parquet")
    default_data_dir = env_path(
        "CG_OUTPUT_ROOT", "stage03_kodcode_dapo_diagnosis", "rl10k", "data"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=default_source, required=default_source is None
    )
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir, required=default_data_dir is None
    )
    parser.add_argument(
        "--reward-path", type=Path, default=STAGE_ROOT / "shared/kodcode_reward.py"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Record failure IDs without returning non-zero (for rebuild loops).",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--exec-timeout", type=float, default=30.0)
    parser.add_argument("--per-test-timeout", type=float, default=2.0)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.data_dir / "audit/exact_reward_preflight.json"
    return args


def load_reward(path: Path):
    spec = importlib.util.spec_from_file_location("diagnose_dapo_reward_preflight", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    reward = load_reward(args.reward_path)
    source = pd.read_parquet(args.source)
    source = source[source["source_dataset"].eq("KodCode-Light-RL-10K")]
    answers = dict(zip(source["id"].astype(str), source["answer"].astype(str), strict=True))
    rows = (
        pq.read_table(args.data_dir / "train.parquet").to_pylist()
        + pq.read_table(args.data_dir / "val.parquet").to_pylist()
    )

    def verify(row):
        problem_id = row["extra_info"]["problem_id"]
        result = reward.compute_score(
            row["data_source"],
            answers[problem_id],
            row["reward_model"]["ground_truth"],
            exec_timeout=args.exec_timeout,
            per_test_timeout=args.per_test_timeout,
            memory_limit_gib=2.0,
            output_limit_mib=16.0,
        )
        return {
            "problem_id": problem_id,
            "subset": row["extra_info"]["subset"],
            "difficulty": row["extra_info"]["difficulty"],
            **result,
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(verify, rows), start=1):
            results.append(result)
            if index % 500 == 0:
                print(f"[reward-preflight] {index}/{len(rows)}", flush=True)
    failures = [row for row in results if row["passed"] != 1.0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "rows": len(results),
        "full_pass": len(results) - len(failures),
        "failed": len(failures),
        "failure_ids": [row["problem_id"] for row in failures],
        "gpu_training_started": False,
        "preflight_config": {
            "workers": args.workers,
            "exec_timeout_seconds": args.exec_timeout,
            "per_test_timeout_seconds": args.per_test_timeout,
            "memory_limit_gib": 2.0,
            "output_limit_mib": 16.0,
        },
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
