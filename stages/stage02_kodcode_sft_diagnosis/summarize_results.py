#!/usr/bin/env python3
"""Create the final Markdown/CSV summary after the unified job finishes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from select_paper_rows import select_paper_rows


TASKS = [
    "arc_challenge",
    "finqa",
    "humaneval",
    "legalbench",
    "math500_medium",
    "mbpp_plus",
    "medcalc",
    "math500_high_level",
    "gsm8k",
    "mbpp_simple",
    "scienceqa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def metric(metrics: dict[str, Any], task: str) -> float:
    for section in ("main", "hard", "diagnostic"):
        value = metrics.get(section, {}).get(task)
        if value:
            return float(value.get("accuracy", value.get("pass_at_1")))
    raise KeyError(task)


def main() -> None:
    args = parse_args()
    formal_root = args.work_dir / "results/formal"
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in formal_root.iterdir() if path.is_dir()):
        for mode_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            base_path = mode_dir / "base/metrics.json"
            if not base_path.is_file():
                continue
            base = json.loads(base_path.read_text(encoding="utf-8"))
            expected_prompt_mode = "chat" if mode_dir.name == "native" else "plain"
            if base.get("prompt_mode") != expected_prompt_mode:
                raise ValueError(f"Base prompt mode mismatch in {mode_dir}")
            if base.get("scoreable_dataset_count") != 11:
                raise ValueError(f"non-standard dataset count in {mode_dir}")
            for checkpoint in ("half_epoch", "final"):
                sft_path = mode_dir / checkpoint / "metrics.json"
                if not sft_path.is_file():
                    continue
                sft = json.loads(sft_path.read_text(encoding="utf-8"))
                if sft.get("prompt_mode") != expected_prompt_mode:
                    raise ValueError(f"{checkpoint} prompt mode mismatch in {mode_dir}")
                if sft.get("scoreable_dataset_count") != 11:
                    raise ValueError(f"non-standard dataset count in {mode_dir}/{checkpoint}")
                row: dict[str, Any] = {
                    "model": model_dir.name,
                    "prompt_mode": mode_dir.name,
                    "checkpoint": checkpoint,
                    "base_overall": float(base["scoreable_overall_mean"]),
                    "sft_overall": float(sft["scoreable_overall_mean"]),
                }
                row["overall_delta"] = row["sft_overall"] - row["base_overall"]
                for task in TASKS:
                    row[f"base_{task}"] = metric(base, task)
                    row[f"sft_{task}"] = metric(sft, task)
                    row[f"delta_{task}"] = row[f"sft_{task}"] - row[f"base_{task}"]
                rows.append(row)

    csv_path = args.work_dir / "results/results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paper_rows = select_paper_rows(rows)
        paper_csv = args.work_dir / "results/paper_rows.csv"
        with paper_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paper_rows[0]))
            writer.writeheader()
            writer.writerows(paper_rows)

    lines = [
        "# SFT diagnosis results",
        "",
        "All values below use the frozen 37,881-row KodCode 4o/R1 clean set and the 11-task formal evaluator.",
        "Hyperparameters are fixed. No validation-based checkpoint selection is used; both checkpoints are reported.",
        "",
        "| Model | Mode | Checkpoint | Base overall | SFT overall | Delta | HumanEval delta | MBPP+ delta | MBPP Simple delta |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {prompt_mode} | {checkpoint} | {base_overall:.4f} | {sft_overall:.4f} | {overall_delta:+.4f} | "
            "{delta_humaneval:+.4f} | {delta_mbpp_plus:+.4f} | {delta_mbpp_simple:+.4f} |".format(**row)
        )
    lines += ["", f"Detailed machine-readable results: `{csv_path}`", ""]
    (args.work_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
