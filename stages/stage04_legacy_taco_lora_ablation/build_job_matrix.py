#!/usr/bin/env python3
"""Build the portable Stage 04 LoRA-SFT training matrix.

The default manifest contains exactly 2 models x 9 arms x 3 seeds = 54 jobs.
It records paths and counts only; training examples and example identifiers are
never embedded in the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path


DEFAULT_MODELS = ("Qwen2.5-7B-Instruct", "Qwen3-8B-Base")
DEFAULT_SEEDS = (20260603, 20260604, 20260605)
DEFAULT_ARMS = (
    "full_sft",
    "full_sft_14043",
    "no_math_number_theory",
    "no_data_structure",
    "no_dynamic_programming",
    "no_greedy_search",
    "no_implementation_simulation",
    "no_graph",
    "no_other_algorithm",
)
FULL_ROWS = 21_879
MATCHED_ROWS = 14_043
EXPERIMENT = "stage2_taco_sft_clean_prompt"


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def unique(values: list[object], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates: {values}")


def parse_args() -> argparse.Namespace:
    data_root = env_path("CG_DATA_ROOT")
    model_root = env_path("CG_MODEL_ROOT")
    output_root = env_path("CG_OUTPUT_ROOT")
    work_root = env_path("CG_WORK_ROOT")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-root", type=Path, default=data_root, required=data_root is None)
    parser.add_argument("--model-root", type=Path, default=model_root, required=model_root is None)
    parser.add_argument("--output-root", type=Path, default=output_root, required=output_root is None)
    parser.add_argument("--work-root", type=Path, default=work_root)
    parser.add_argument("--output", type=Path, help="Write JSON here; stdout is used when omitted.")
    args = parser.parse_args()
    if args.work_root is None:
        args.work_root = args.output_root / ".work"
    unique(args.models, "models")
    unique(args.arms, "arms")
    unique(args.seeds, "seeds")
    unknown_arms = sorted(set(args.arms) - set(DEFAULT_ARMS))
    if unknown_arms:
        raise ValueError(f"unknown Stage 04 arms: {unknown_arms}")
    return args


def main() -> None:
    args = parse_args()
    jobs = []
    for seed, model, arm in product(args.seeds, args.models, args.arms):
        job_id = f"seed{seed}__{model}__{arm}"
        jobs.append(
            {
                "job_id": job_id,
                "seed": seed,
                "model": model,
                "arm": arm,
                "expected_rows": FULL_ROWS if arm == "full_sft" else MATCHED_ROWS,
                "model_path": str(args.model_root / model),
                "train_file": str(args.data_root / EXPERIMENT / f"{arm}.jsonl"),
                "adapter_dir": str(
                    args.output_root
                    / "checkpoints"
                    / "sft_lora"
                    / EXPERIMENT
                    / f"seed_{seed}"
                    / model
                    / arm
                ),
                "log_dir": str(
                    args.work_root / "logs" / EXPERIMENT / f"seed_{seed}" / model / arm
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "experiment": "stage04_legacy_taco_lora_ablation",
        "matrix_formula": f"{len(args.models)} models x {len(args.arms)} arms x {len(args.seeds)} seeds",
        "job_count": len(jobs),
        "models": args.models,
        "arms": args.arms,
        "seeds": args.seeds,
        "jobs": jobs,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)


if __name__ == "__main__":
    main()
