#!/usr/bin/env python3
"""Strictly select the best saved nonzero DAPO step using validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run containing validation/*.jsonl.")
    parser.add_argument("--adapter-root", type=Path, required=True, help="Root containing adapters/step_XXXXXX.")
    parser.add_argument("--metric", choices=("pass_frac", "passed", "score"), default="pass_frac")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-path", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_for_step(root: Path, step: int) -> Path:
    candidates = (root / "adapters" / f"step_{step:06d}", root / f"step_{step:06d}")
    for candidate in candidates:
        if (candidate / "adapter_model.safetensors").is_file() or (candidate / "adapter_model.bin").is_file():
            return candidate
    raise FileNotFoundError(f"no saved adapter for validation step {step} under {root}")


def numeric(row: dict[str, Any], name: str, path: Path) -> float:
    value = row.get(name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{path}: validation row is missing numeric {name!r}")
    return float(value)


def main() -> None:
    args = parse_args()
    validation_dir = args.run_dir / "validation"
    if not validation_dir.is_dir():
        raise FileNotFoundError(f"validation directory not found: {validation_dir}")
    paths = sorted(
        (path for path in validation_dir.glob("*.jsonl") if path.stem.isdigit() and int(path.stem) > 0),
        key=lambda path: int(path.stem),
    )
    if not paths:
        raise FileNotFoundError(f"no nonzero validation step JSONL under {validation_dir}")

    records: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for path in paths:
        step = int(path.stem)
        if step in seen_steps:
            raise ValueError(f"duplicate validation step: {step}")
        seen_steps.add(step)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"empty or malformed validation file: {path}")
        adapter = adapter_for_step(args.adapter_root, step)
        records.append(
            {
                "step": step,
                "adapter": str(adapter.resolve()),
                "validation_path": str(path.resolve()),
                "validation_sha256": sha256(path),
                "n": len(rows),
                "pass_frac": statistics.fmean(numeric(row, "pass_frac", path) for row in rows),
                "passed": statistics.fmean(numeric(row, "passed", path) for row in rows),
                "score": statistics.fmean(numeric(row, "score", path) for row in rows),
            }
        )

    best = max(records, key=lambda row: (row[args.metric], row["passed"], row["step"]))
    result = {
        "schema_version": 1,
        "selection_policy": (
            f"maximum mean {args.metric} over saved nonzero validation steps; "
            "tie-break by mean passed then later step; test metrics are never consulted"
        ),
        "run_dir": str(args.run_dir.resolve()),
        "adapter_root": str(args.adapter_root.resolve()),
        "selected": best,
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.print_path:
        print(best["adapter"])
    else:
        print(json.dumps({"selected_step": best["step"], "adapter": best["adapter"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
