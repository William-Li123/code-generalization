#!/usr/bin/env python3
"""Select a DAPO adapter by held-out validation; fail closed by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--expected-validation-rows", type=int, default=100)
    parser.add_argument(
        "--exploratory-fallback-latest",
        action="store_true",
        help="Explicitly allow latest_adapter.txt when no valid validation result exists.",
    )
    args = parser.parse_args()
    candidates: list[dict[str, object]] = []
    for run_dir in sorted(args.output_root.glob(f"{args.model_key}-*")):
        validation_dir = run_dir / "validation"
        if not validation_dir.is_dir():
            continue
        for path in validation_dir.glob("*.jsonl"):
            if not path.stem.isdigit() or int(path.stem) <= 0:
                continue
            step = int(path.stem)
            adapter = args.checkpoint_root / "adapters" / f"step_{step:06d}"
            if not (
                (adapter / "adapter_model.safetensors").is_file()
                and (adapter / "adapter_config.json").is_file()
            ):
                continue
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not rows:
                continue
            if len(rows) != args.expected_validation_rows:
                raise ValueError(
                    f"validation row-count mismatch in {path}: "
                    f"expected {args.expected_validation_rows}, found {len(rows)}"
                )
            candidates.append(
                {
                    "step": step,
                    "adapter": adapter,
                    "pass_frac": mean([float(row.get("pass_frac") or 0) for row in rows]),
                    "passed": mean([float(row.get("passed") or 0) for row in rows]),
                    "score": mean([float(row.get("score") or 0) for row in rows]),
                    "n": len(rows),
                    "validation_file": path,
                }
            )
    if candidates:
        best = max(
            candidates,
            key=lambda row: (
                row["passed"],
                row["score"],
                row["pass_frac"],
                row["step"],
            ),
        )
        reason = "best_validation_full_pass_rate"
    else:
        if not args.exploratory_fallback_latest:
            raise RuntimeError(
                "no adapter has a complete held-out validation file; refusing to "
                "select latest_adapter.txt in the formal fail-closed policy"
            )
        latest_file = args.checkpoint_root / "latest_adapter.txt"
        if not latest_file.is_file():
            raise FileNotFoundError(latest_file)
        latest = Path(latest_file.read_text(encoding="utf-8").strip())
        if not latest.is_absolute():
            latest = args.checkpoint_root / latest
        best = {
            "step": int(latest.name.removeprefix("step_")) if latest.name.startswith("step_") else -1,
            "adapter": latest,
            "pass_frac": None,
            "passed": None,
            "score": None,
            "n": None,
            "validation_file": None,
        }
        reason = "exploratory_fallback_latest_adapter"
    if not (Path(best["adapter"]) / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(Path(best["adapter"]) / "adapter_model.safetensors")
    args.selection_file.parent.mkdir(parents=True, exist_ok=True)
    args.selection_file.write_text(
        json.dumps(
            {
                "model_key": args.model_key,
                "reason": reason,
                **{key: str(value) if isinstance(value, Path) else value for key, value in best.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(best["adapter"])


if __name__ == "__main__":
    main()
