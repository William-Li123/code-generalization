#!/usr/bin/env python3
"""Serially repeat exact-reward checks that failed only in concurrent preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


STAGE_ROOT = Path(__file__).resolve().parents[1]
TARGET_IDS = (
    "rl10k_r1::Filter_60220_I",
    "rl10k_r1::Filter_1152_I",
    "rl10k_r1::Filter_80484_I",
)
REPEATS = 5


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
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.data_dir / "audit/transient_reward_serial_recheck.json"
    return args


def load_reward(path: Path):
    spec = importlib.util.spec_from_file_location("diagnose_dapo_reward_recheck", path)
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
    selected = {
        row["extra_info"]["problem_id"]: row
        for row in rows
        if row["extra_info"]["problem_id"] in TARGET_IDS
    }
    if set(selected) != set(TARGET_IDS):
        raise RuntimeError(f"missing target IDs: {set(TARGET_IDS) - set(selected)}")

    results = {}
    for problem_id in TARGET_IDS:
        row = selected[problem_id]
        attempts = []
        for _ in range(args.repeats):
            attempts.append(
                reward.compute_score(
                    row["data_source"],
                    answers[problem_id],
                    row["reward_model"]["ground_truth"],
                    exec_timeout=30.0,
                    per_test_timeout=2.0,
                    memory_limit_gib=2.0,
                    output_limit_mib=16.0,
                )
            )
        results[problem_id] = attempts

    all_pass = all(
        attempt["passed"] == 1.0
        for attempts in results.values()
        for attempt in attempts
    )
    report = {
        "reason": "three concurrent-preflight-only failures",
        "mode": "serial exact reward",
        "repeats_per_problem": args.repeats,
        "problems": len(TARGET_IDS),
        "attempts": len(TARGET_IDS) * args.repeats,
        "all_pass": all_pass,
        "results": results,
        "gpu_training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
