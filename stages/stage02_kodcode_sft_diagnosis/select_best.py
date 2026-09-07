#!/usr/bin/env python3
"""Select a pilot profile or full adapter without consulting public benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="label=eval_dir=adapter_path")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for row in (
            json.loads(line) for line in (path / "records.jsonl").open(encoding="utf-8") if line.strip()
        )
    }


def main() -> None:
    args = parse_args()
    baseline = read_records(args.baseline_dir)
    baseline_summary = json.loads((args.baseline_dir / "summary.json").read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for spec in args.candidate:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"invalid candidate: {spec}")
        label, eval_dir_text, adapter_path = parts
        eval_dir = Path(eval_dir_text)
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        records = read_records(eval_dir)
        common = sorted(set(baseline) & set(records))
        pass_to_fail = sum(baseline[key]["passed"] and not records[key]["passed"] for key in common)
        fail_to_pass = sum(not baseline[key]["passed"] and records[key]["passed"] for key in common)
        candidate = {
            "label": label,
            "eval_dir": str(eval_dir),
            "adapter_path": adapter_path,
            "n_common": len(common),
            "passed": int(summary["passed"]),
            "pass_rate": float(summary["pass_rate"]),
            "mixed_macro_accuracy": float(summary["mixed_macro_accuracy"]),
            "code_macro_accuracy": float(summary["code_macro_accuracy"]),
            "components": summary["components"],
            "mean_test_pass_fraction": float(summary["mean_test_pass_fraction"]),
            "pass_to_fail": pass_to_fail,
            "fail_to_pass": fail_to_pass,
            "net_transition": fail_to_pass - pass_to_fail,
            "truncation_rate": float(summary["truncation_rate"]),
        }
        candidates.append(candidate)
    # This is a code-SFT diagnosis, so checkpoint selection is code-first.
    # Mathematics remains a secondary guard against broad capability loss.
    def ranking_key(item: dict[str, Any]) -> tuple[float, ...]:
        return (
            item["code_macro_accuracy"],
            item["mixed_macro_accuracy"],
            -item["pass_to_fail"],
            item["fail_to_pass"],
            item["mean_test_pass_fraction"],
            -item["truncation_rate"],
        )

    candidates.sort(key=ranking_key, reverse=True)
    baseline_rank = (
        float(baseline_summary["code_macro_accuracy"]),
        float(baseline_summary["mixed_macro_accuracy"]),
        0,
        0,
        float(baseline_summary["mean_test_pass_fraction"]),
        -float(baseline_summary["truncation_rate"]),
    )
    selected_beats_baseline = ranking_key(candidates[0]) > baseline_rank
    result = {
        "selection_data": "private verified code plus disjoint public validation splits",
        "formal_test_sets_used_for_selection": False,
        "public_validation_splits_used_for_selection": True,
        "selection_components": "private code, held-out non-overlap MBPP validation, MMLU math validation",
        "ranking_rule": "code macro accuracy desc, three-component macro accuracy desc, pass_to_fail asc, fail_to_pass desc, fractional pass desc, truncation asc",
        "baseline": {
            "passed": int(baseline_summary["passed"]),
            "pass_rate": float(baseline_summary["pass_rate"]),
            "mixed_macro_accuracy": float(baseline_summary["mixed_macro_accuracy"]),
            "code_macro_accuracy": float(baseline_summary["code_macro_accuracy"]),
            "components": baseline_summary["components"],
            "mean_test_pass_fraction": float(baseline_summary["mean_test_pass_fraction"]),
            "truncation_rate": float(baseline_summary["truncation_rate"]),
        },
        "selected": candidates[0],
        "selected_beats_baseline": selected_beats_baseline,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output.parent / "best_adapter_path.txt").write_text(
        candidates[0]["adapter_path"] + "\n", encoding="utf-8"
    )
    (args.output.parent / "selected_beats_baseline.txt").write_text(
        str(selected_beats_baseline).lower() + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
