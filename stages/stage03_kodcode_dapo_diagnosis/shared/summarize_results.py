#!/usr/bin/env python3
"""Collect the eight matching-mode base/DAPO formal metrics files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


MODELS = ["qwen25", "qwen3", "llama31", "gemma2"]
STAGES = ["base", "dapo_best"]
TASKS = [
    "arc_challenge",
    "scienceqa",
    "legalbench",
    "gsm8k",
    "math500_medium",
    "math500_high_level",
    "finqa",
    "medcalc",
    "humaneval",
    "mbpp_plus",
    "mbpp_simple",
]


def parse_args() -> argparse.Namespace:
    output_root = os.environ.get("CG_OUTPUT_ROOT")
    default_work_dir = (
        Path(output_root).expanduser() / "stage03_kodcode_dapo_diagnosis/rl10k"
        if output_root
        else None
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=default_work_dir,
        required=default_work_dir is None,
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.work_dir / "results/summary.csv"
    return args


def metric(payload: dict, task: str) -> float:
    for suite in ("main", "hard", "diagnostic"):
        value = payload.get(suite, {}).get(task)
        if isinstance(value, dict):
            for key in ("accuracy", "pass_rate", "score"):
                if key in value:
                    return float(value[key])
    raise KeyError(task)


def main() -> None:
    args = parse_args()
    rows = []
    for model in args.models:
        for stage in STAGES:
            path = args.work_dir / "results/formal" / model / stage / "metrics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = {task: metric(payload, task) for task in TASKS}
            if payload.get("scoreable_dataset_count") != 11:
                raise RuntimeError(f"incomplete formal metrics: {path}")
            if task_n := payload.get("main", {}).get("gsm8k", {}).get("n"):
                if int(task_n) != 1319:
                    raise RuntimeError(f"non-full GSM8K result: {path}: n={task_n}")
            rows.append(
                {
                    "model": model,
                    "stage": stage,
                    "overall": sum(values.values()) / len(values),
                    **values,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
