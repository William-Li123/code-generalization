#!/usr/bin/env python3
"""Write compact absolute and delta-vs-base CSVs for one Stage 1 model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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
SUITES = {
    "arc_challenge": "main",
    "finqa": "main",
    "humaneval": "main",
    "legalbench": "main",
    "math500_medium": "main",
    "mbpp_plus": "main",
    "medcalc": "main",
    "math500_high_level": "hard",
    "gsm8k": "diagnostic",
    "mbpp_simple": "diagnostic",
    "scienceqa": "diagnostic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def metric_value(metric: dict[str, Any]) -> float:
    value = metric.get("accuracy", metric.get("pass_at_1", metric.get("score")))
    if value is None:
        raise ValueError(f"metric has no supported score key: {metric.keys()}")
    return float(value)


def row_from_metrics(name: str, path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    row: dict[str, Any] = {"checkpoint": name}
    values = []
    for task in TASKS:
        metric = data[SUITES[task]][task]
        value = metric_value(metric)
        row[task] = value
        values.append(value)
    row["overall"] = sum(values) / len(values)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["checkpoint", "overall", *TASKS]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.eval_root.glob("*/metrics.json")):
        rows.append(row_from_metrics(path.parent.name, path))
    if not rows:
        raise ValueError(f"no metrics.json found under {args.eval_root}")
    rows.sort(key=lambda row: (row["checkpoint"] != "base", row["checkpoint"]))
    base = next((row for row in rows if row["checkpoint"] == "base"), None)
    if base is None:
        raise ValueError("base metrics are required")
    deltas = []
    for row in rows:
        delta = {"checkpoint": row["checkpoint"]}
        for key in ["overall", *TASKS]:
            delta[key] = float(row[key]) - float(base[key])
        deltas.append(delta)
    write_csv(args.output_dir / "absolute.csv", rows)
    write_csv(args.output_dir / "delta_vs_base.csv", deltas)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rows": len(rows),
                "tasks": TASKS,
                "absolute": "absolute.csv",
                "delta_vs_base": "delta_vs_base.csv",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
