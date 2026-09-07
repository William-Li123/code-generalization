#!/usr/bin/env python3
"""Build the archived 36-cell Stage 06 full-SFT training matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


MODELS = ("Qwen2.5-7B-Instruct", "Qwen3-8B-Base")
ARMS = (
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
ARCHIVED_TRAIN_SEEDS = (20260603, 20260604)


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def parse_args() -> argparse.Namespace:
    data_root = env_path("CG_DATA_ROOT")
    model_root = env_path("CG_MODEL_ROOT")
    output_root = env_path("CG_OUTPUT_ROOT")
    work_root = env_path("CG_WORK_ROOT")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=data_root, required=data_root is None)
    parser.add_argument("--model-root", type=Path, default=model_root, required=model_root is None)
    parser.add_argument("--output-root", type=Path, default=output_root, required=output_root is None)
    parser.add_argument("--work-root", type=Path, default=work_root)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(ARCHIVED_TRAIN_SEEDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.work_root is None:
        args.work_root = args.output_root / ".work"
    if tuple(args.seeds) != ARCHIVED_TRAIN_SEEDS:
        parser.error(f"formal archived matrix requires seeds {ARCHIVED_TRAIN_SEEDS}")
    return args


def main() -> None:
    args = parse_args()
    jobs = []
    for seed in args.seeds:
        exp_name = f"stage4_taco_full_sft_seed_{seed}"
        for model in MODELS:
            for arm in ARMS:
                jobs.append(
                    {
                        "job_id": f"seed{seed}__{model}__{arm}",
                        "training_seed": seed,
                        "model": model,
                        "arm": arm,
                        "model_path": str(args.model_root / model),
                        "train_file": str(args.data_root / "stage2_taco_sft_clean_prompt" / f"{arm}.jsonl"),
                        "checkpoint_dir": str(args.output_root / "checkpoints" / "sft_full" / exp_name / model / arm),
                        "log_dir": str(args.work_root / "logs" / exp_name / model / arm),
                    }
                )
    manifest = {
        "schema_version": 1,
        "experiment": "stage06_legacy_full_sft_robustness",
        "archived_training_seeds": list(args.seeds),
        "paper_reported_seeds": [20260604, 20260605],
        "legacy_evaluation_seed": 20260605,
        "models": list(MODELS),
        "arms": list(ARMS),
        "job_count": len(jobs),
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
