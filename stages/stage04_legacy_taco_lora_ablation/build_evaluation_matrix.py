#!/usr/bin/env python3
"""Build the Stage 04 base-once plus adapter evaluation matrix."""

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
PAPER_TASKS = (
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
)
TRAINING_EXPERIMENT = "stage2_taco_sft_clean_prompt"
STAGE_NAME = "stage04_legacy_taco_lora_ablation"


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def ensure_unique(values: list[object], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates: {values}")


def parse_args() -> argparse.Namespace:
    data_test_root = env_path("CG_DATA_TEST_ROOT")
    model_root = env_path("CG_MODEL_ROOT")
    output_root = env_path("CG_OUTPUT_ROOT")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--base-eval-seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument(
        "--data-test-root",
        type=Path,
        default=data_test_root,
        required=data_test_root is None,
    )
    parser.add_argument("--model-root", type=Path, default=model_root, required=model_root is None)
    parser.add_argument("--output-root", type=Path, default=output_root, required=output_root is None)
    parser.add_argument("--output", type=Path, help="Write JSON here; stdout is used when omitted.")
    args = parser.parse_args()
    ensure_unique(args.models, "models")
    ensure_unique(args.arms, "arms")
    ensure_unique(args.seeds, "seeds")
    unknown_arms = sorted(set(args.arms) - set(DEFAULT_ARMS))
    if unknown_arms:
        raise ValueError(f"unknown Stage 04 arms: {unknown_arms}")
    return args


def main() -> None:
    args = parse_args()
    eval_root = args.output_root / "evaluation" / STAGE_NAME
    adapter_root = (
        args.output_root / "checkpoints" / "sft_lora" / TRAINING_EXPERIMENT
    )
    jobs: list[dict[str, object]] = []

    # A stochastic base evaluation is deliberately run once per model, not once
    # per training seed. Downstream aggregation reuses this single base record.
    for model in args.models:
        jobs.append(
            {
                "job_id": f"base__{model}",
                "kind": "base",
                "model": model,
                "arm": "base",
                "training_seed": None,
                "evaluation_seed": args.base_eval_seed,
                "base_model_path": str(args.model_root / model),
                "adapter_path": None,
                "data_test_root": str(args.data_test_root),
                "output_dir": str(eval_root / "base" / model),
                "tasks": "scoreable",
                "limit": 0,
            }
        )

    for seed, model, arm in product(args.seeds, args.models, args.arms):
        jobs.append(
            {
                "job_id": f"seed{seed}__{model}__{arm}",
                "kind": "adapter",
                "model": model,
                "arm": arm,
                "training_seed": seed,
                "evaluation_seed": seed,
                "base_model_path": str(args.model_root / model),
                "adapter_path": str(adapter_root / f"seed_{seed}" / model / arm),
                "data_test_root": str(args.data_test_root),
                "output_dir": str(eval_root / f"seed_{seed}" / model / arm),
                "tasks": "scoreable",
                "limit": 0,
            }
        )

    for index, job in enumerate(jobs):
        job["job_index"] = index
    base_jobs = sum(job["kind"] == "base" for job in jobs)
    adapter_jobs = sum(job["kind"] == "adapter" for job in jobs)
    manifest = {
        "schema_version": 1,
        "experiment": STAGE_NAME,
        "evaluator": "../../evaluation/legacy_suite.py",
        "tasks_alias": "scoreable",
        "paper_task_count": len(PAPER_TASKS),
        "paper_tasks": list(PAPER_TASKS),
        "base_policy": "one evaluation per model; reuse across all adapter training seeds",
        "base_evaluation_seed": args.base_eval_seed,
        "models": args.models,
        "arms": args.arms,
        "training_seeds": args.seeds,
        "base_job_count": base_jobs,
        "adapter_job_count": adapter_jobs,
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
