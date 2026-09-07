#!/usr/bin/env python3
"""Rebuild the paper's legacy-vs-main base calibration table (Table 7)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL = "Qwen2.5-7B-Instruct"
TASKS = {
    "HumanEval": "humaneval",
    "MBPP-Simple": "mbpp_simple",
    "MBPP+": "mbpp_plus",
    "GSM8K": "gsm8k",
    "MATH-Med": "math500_medium",
    "MATH-High": "math500_high_level",
    "FinQA": "finqa",
    "MedCalc": "medcalc",
    "ARC-C": "arc_challenge",
    "LegalBench": "legalbench",
    "ScienceQA": "scienceqa",
}


def load_main(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["model"] == MODEL and row["arm"] == "base"]
    if len(selected) != 1:
        raise ValueError(f"expected one main Qwen2.5 Base row in {path}, got {len(selected)}")
    row = selected[0]
    return {display: float(row[column]) for display, column in TASKS.items()}


def load_legacy(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["model"] == MODEL and row["arm"] == "base"]
    if len(selected) != len(TASKS) + 1:  # eleven tasks plus Overall
        raise ValueError(f"legacy Base rows are incomplete in {path}: {len(selected)}")
    values = {row["dataset"]: float(row["score"]) for row in selected}
    missing = set(TASKS) - set(values)
    if missing:
        raise ValueError(f"legacy Base tasks are missing: {sorted(missing)}")
    return {display: values[display] for display in TASKS}


def markdown(rows: list[dict[str, float | str]]) -> str:
    lines = [
        "| task | legacy_base | main_base | main_minus_legacy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['legacy_base']:.2f} | {row['main_base']:.2f} | "
            f"{row['main_minus_legacy']:+.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-package",
        type=Path,
        required=True,
        help="Stage-01 package root containing csv/sft_lora_average.csv.",
    )
    parser.add_argument(
        "--legacy-package",
        type=Path,
        required=True,
        help="Output of package_legacy_results.py containing data/dapo_scores_long.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    main_values = load_main(args.main_package.resolve() / "csv" / "sft_lora_average.csv")
    legacy_values = load_legacy(
        args.legacy_package.resolve() / "data" / "dapo_scores_long.csv"
    )
    rows = [
        {
            "task": task,
            "legacy_base": legacy_values[task],
            "main_base": main_values[task],
            "main_minus_legacy": main_values[task] - legacy_values[task],
        }
        for task in TASKS
    ]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "table7_calibration.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "table7_calibration.md").write_text(markdown(rows), encoding="utf-8")
    (output / "table7_manifest.json").write_text(
        json.dumps(
            {
                "contract": "same_untrained_qwen25_base_across_two_evaluation_harnesses",
                "main_input": str(
                    args.main_package.resolve() / "csv" / "sft_lora_average.csv"
                ),
                "legacy_input": str(
                    args.legacy_package.resolve() / "data" / "dapo_scores_long.csv"
                ),
                "rows": len(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
