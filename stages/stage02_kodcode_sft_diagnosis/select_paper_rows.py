#!/usr/bin/env python3
"""Select the frozen Appendix-B SFT rows from the full prompt-mode sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


PAPER_PROMPT_MODE = {
    "qwen25": "native",
    "qwen3": "plain",
    "llama31": "native",
    "gemma2": "native",
}
PAPER_CHECKPOINTS = ("half_epoch", "final")


def select_paper_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["prompt_mode"]), str(row["checkpoint"]))
        if key in indexed:
            raise ValueError(f"duplicate Stage-02 result row: {key}")
        indexed[key] = row
    selected: list[dict[str, Any]] = []
    missing: list[tuple[str, str, str]] = []
    for model, mode in PAPER_PROMPT_MODE.items():
        for checkpoint in PAPER_CHECKPOINTS:
            key = (model, mode, checkpoint)
            if key in indexed:
                selected.append(indexed[key])
            else:
                missing.append(key)
    if missing:
        raise ValueError(f"missing frozen Appendix-B SFT rows: {missing}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    with args.results_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = select_paper_rows(csv.DictReader(handle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                {
                    "contract": "appendix_b_stage02_paper_rows_v1",
                    "prompt_mode_by_model": PAPER_PROMPT_MODE,
                    "checkpoints": list(PAPER_CHECKPOINTS),
                    "rows": len(rows),
                    "source_results": str(args.results_csv),
                    "output": str(args.output),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(args.output)


if __name__ == "__main__":
    main()
