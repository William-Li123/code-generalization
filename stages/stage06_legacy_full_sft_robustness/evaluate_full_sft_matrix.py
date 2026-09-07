#!/usr/bin/env python3
"""Evaluate the 36 archived full-SFT checkpoints plus two Base references."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


TASK_N = {
    "arc_challenge": 1172,
    "finqa": 1133,
    "humaneval": 164,
    "legalbench": 1689,
    "math500_medium": 105,
    "mbpp_plus": 378,
    "medcalc": 1100,
    "math500_high_level": 262,
    "mbpp_simple": 257,
    "scienceqa": 2224,
}
SUITE = {
    **{task: "main" for task in ("arc_challenge", "finqa", "humaneval", "legalbench", "math500_medium", "mbpp_plus", "medcalc")},
    "math500_high_level": "hard",
    **{task: "diagnostic" for task in ("mbpp_simple", "scienceqa")},
}
CONTRACT = "appendix_a_10_no_gsm8k"
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
TRAINING_SEEDS = (20260603, 20260604)


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def parse_args() -> argparse.Namespace:
    output_root = env_path("CG_OUTPUT_ROOT")
    data_test_root = env_path("CG_DATA_TEST_ROOT")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=output_root, required=output_root is None)
    parser.add_argument("--data-test-root", type=Path, default=data_test_root, required=data_test_root is None)
    parser.add_argument("--evaluation-seed", type=int, default=20260605)
    parser.add_argument("--manifest-output", type=Path)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--all", action="store_true")
    execution.add_argument("--job-index", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_training_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if (
        payload.get("schema_version") != 1
        or payload.get("job_count") != 36
        or not isinstance(jobs, list)
        or len(jobs) != 36
    ):
        raise ValueError("formal Stage 06 training matrix must contain exactly 36 jobs")
    cells = {(int(job["training_seed"]), job["model"], job["arm"]) for job in jobs}
    expected = {(seed, model, arm) for seed in TRAINING_SEEDS for model in MODELS for arm in ARMS}
    if cells != expected:
        raise ValueError(
            f"Stage 06 training matrix mismatch; missing={sorted(expected - cells)}, extra={sorted(cells - expected)}"
        )
    return jobs


def build_jobs(training: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    root = output_root / "stage06_legacy_full_sft_robustness" / "evaluation" / CONTRACT
    model_paths: dict[str, str] = {}
    for job in training:
        model_paths.setdefault(str(job["model"]), str(job["model_path"]))
    jobs: list[dict[str, Any]] = []
    for model in MODELS:
        jobs.append(
            {
                "kind": "base",
                "label": f"{model}__base",
                "model": model,
                "arm": "base",
                "training_seed": None,
                "checkpoint_path": model_paths[model],
                "output_dir": str(root / "base" / f"{model}__base"),
            }
        )
    for job in training:
        seed = int(job["training_seed"])
        model = str(job["model"])
        arm = str(job["arm"])
        jobs.append(
            {
                "kind": "full_sft",
                "label": f"{model}__{arm}",
                "model": model,
                "arm": arm,
                "training_seed": seed,
                "checkpoint_path": str(job["checkpoint_dir"]),
                "output_dir": str(root / f"train_seed_{seed}" / f"{model}__{arm}"),
            }
        )
    return jobs


def validate_metrics(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data.get("tasks_requested", [])) != set(TASK_N):
        raise ValueError(f"{path}: expected exactly the Appendix-A ten-task suite")
    for task, expected_n in TASK_N.items():
        metric = data.get(SUITE[task], {}).get(task)
        if not isinstance(metric, dict) or metric.get("n") != expected_n:
            raise ValueError(f"{path}: {task} expected n={expected_n}, got {metric}")


def run_job(job: dict[str, Any], args: argparse.Namespace, evaluator: Path) -> None:
    output_dir = Path(str(job["output_dir"]))
    metrics = output_dir / "metrics.json"
    if metrics.exists() and not args.overwrite:
        validate_metrics(metrics)
        return
    subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--data-root",
            str(args.data_test_root),
            "--model-name",
            str(job["label"]),
            "--base-model-path",
            str(job["checkpoint_path"]),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(args.evaluation_seed),
            "--tasks",
            ",".join(TASK_N),
            "--skip-healthbench-generation",
        ],
        check=True,
    )
    validate_metrics(metrics)


def main() -> None:
    args = parse_args()
    stage_dir = Path(__file__).resolve().parent
    evaluator = stage_dir.parents[1] / "evaluation" / "legacy_suite.py"
    jobs = build_jobs(load_training_jobs(args.training_matrix), args.output_root)
    if len(jobs) != 38:
        raise RuntimeError(f"expected 38 evaluation jobs, got {len(jobs)}")
    manifest = args.manifest_output or (
        args.output_root / "stage06_legacy_full_sft_robustness" / "evaluation" / "evaluation_job_matrix.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluation_contract": CONTRACT,
                "evaluation_seed": args.evaluation_seed,
                "job_count": len(jobs),
                "jobs": jobs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.job_index is not None and not 0 <= args.job_index < len(jobs):
        raise IndexError(f"--job-index must be in [0, {len(jobs) - 1}]")
    selected = jobs if (args.all or args.validate_only) else ([jobs[args.job_index]] if args.job_index is not None else [])
    for job in selected:
        metrics = Path(str(job["output_dir"])) / "metrics.json"
        if args.validate_only:
            validate_metrics(metrics)
        else:
            run_job(job, args, evaluator)
    print(json.dumps({"manifest": str(manifest), "jobs": len(jobs), "processed": len(selected)}))


if __name__ == "__main__":
    main()
