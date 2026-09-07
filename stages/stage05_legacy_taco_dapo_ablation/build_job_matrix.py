#!/usr/bin/env python3
"""Build the archived Stage 05 complete-reference and matched DAPO matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


MODELS = ("Qwen2.5-7B-Instruct", "Qwen3-8B-Base")
MATCHED_ARMS = (
    "full_dapo_12168",
    "no_math_number_theory",
    "no_data_structure",
    "no_dynamic_programming",
    "no_greedy_search",
    "no_implementation_simulation",
    "no_graph",
    "no_other_algorithm",
)
COMPLETE_SEED = 20260609
MATCHED_SEED = 20260610
COMPLETE_ROWS = 17249
MATCHED_ROWS = 12168


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def parse_args() -> argparse.Namespace:
    data_root = env_path("CG_DATA_ROOT")
    model_root = env_path("CG_MODEL_ROOT")
    output_root = env_path("CG_OUTPUT_ROOT")
    work_root = env_path("CG_WORK_ROOT")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--regime",
        choices=("all", "complete", "matched"),
        default="all",
        help="The default emits two complete references plus the 16 matched jobs.",
    )
    parser.add_argument("--matched-arms", nargs="+", choices=MATCHED_ARMS, default=list(MATCHED_ARMS))
    parser.add_argument("--complete-seed", type=int, default=COMPLETE_SEED)
    parser.add_argument("--matched-seed", type=int, default=MATCHED_SEED)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--data-root", type=Path, default=data_root, required=data_root is None)
    parser.add_argument("--model-root", type=Path, default=model_root, required=model_root is None)
    parser.add_argument("--output-root", type=Path, default=output_root, required=output_root is None)
    parser.add_argument("--work-root", type=Path, default=work_root)
    parser.add_argument("--output", type=Path, help="Write JSON here; stdout is used when omitted.")
    args = parser.parse_args()
    if args.work_root is None:
        args.work_root = args.output_root / ".work"
    if args.train_batch_size <= 0:
        parser.error("--train-batch-size must be positive")
    return args


def model_settings(model: str) -> tuple[int, int]:
    return (3072, 4) if model == "Qwen3-8B-Base" else (2048, 8)


def make_job(
    *,
    model: str,
    arm: str,
    regime: str,
    seed: int,
    expected_rows: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    max_response, micro_batch = model_settings(model)
    job_id = f"{regime}__{model}__{arm}__seed{seed}"
    if regime == "complete":
        data_dir = args.data_root / "stage3_dapo_full_verified"
    else:
        data_dir = args.data_root / "stage3_dapo_ablation_12168" / f"seed_{seed}" / arm
    relative = Path(regime) / f"seed_{seed}" / model / arm
    return {
        "job_id": job_id,
        "regime": regime,
        "model": model,
        "arm": arm,
        "seed": seed,
        "expected_train_rows": expected_rows,
        "train_batch_size": args.train_batch_size,
        "total_steps": math.ceil(expected_rows / args.train_batch_size),
        "model_path": str(args.model_root / model),
        "data_dir": str(data_dir),
        "max_response_length": max_response,
        "micro_batch_size_per_gpu": micro_batch,
        "output_dir": str(args.output_root / "stage05_legacy_taco_dapo_ablation" / relative),
        "work_dir": str(args.work_root / "stage05_legacy_taco_dapo_ablation" / relative),
    }


def main() -> None:
    args = parse_args()
    jobs: list[dict[str, object]] = []
    if args.regime in {"all", "complete"}:
        for model in args.models:
            jobs.append(
                make_job(
                    model=model,
                    arm="full_dapo",
                    regime="complete",
                    seed=args.complete_seed,
                    expected_rows=COMPLETE_ROWS,
                    args=args,
                )
            )
    if args.regime in {"all", "matched"}:
        for model in args.models:
            for arm in args.matched_arms:
                jobs.append(
                    make_job(
                        model=model,
                        arm=arm,
                        regime="matched",
                        seed=args.matched_seed,
                        expected_rows=MATCHED_ROWS,
                        args=args,
                    )
                )

    ids = [str(job["job_id"]) for job in jobs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate Stage 05 job ids")
    manifest = {
        "schema_version": 2,
        "experiment": "stage05_legacy_taco_dapo_ablation",
        "evaluation_contract": "canonical_11_20260719",
        "complete_reference": {
            "seed": args.complete_seed,
            "rows": COMPLETE_ROWS,
            "arms_per_model": 1,
        },
        "matched_attribution": {
            "seed": args.matched_seed,
            "rows_per_arm": MATCHED_ROWS,
            "arms_per_model": len(args.matched_arms),
        },
        "job_count": len(jobs),
        "models": args.models,
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
