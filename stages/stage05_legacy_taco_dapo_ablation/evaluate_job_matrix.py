#!/usr/bin/env python3
"""Select Stage 05 adapters, run the canonical legacy suite, and validate outputs."""

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
    "gsm8k": 1319,
    "mbpp_simple": 257,
    "scienceqa": 2224,
}
SUITE = {
    **{task: "main" for task in ("arc_challenge", "finqa", "humaneval", "legalbench", "math500_medium", "mbpp_plus", "medcalc")},
    "math500_high_level": "hard",
    **{task: "diagnostic" for task in ("gsm8k", "mbpp_simple", "scienceqa")},
}
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


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def parse_args() -> argparse.Namespace:
    output_root = env_path("CG_OUTPUT_ROOT")
    model_root = env_path("CG_MODEL_ROOT")
    data_test_root = env_path("CG_DATA_TEST_ROOT")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=output_root, required=output_root is None)
    parser.add_argument("--model-root", type=Path, default=model_root, required=model_root is None)
    parser.add_argument("--data-test-root", type=Path, default=data_test_root, required=data_test_root is None)
    parser.add_argument("--evaluation-seed", type=int, default=20260610)
    parser.add_argument("--manifest-output", type=Path)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--all", action="store_true")
    execution.add_argument("--job-index", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_training_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if payload.get("schema_version") != 2 or not isinstance(jobs, list) or len(jobs) != payload.get("job_count"):
        raise ValueError("invalid Stage 05 training manifest")
    if len(jobs) != 18:
        raise ValueError(f"formal Stage 05 manifest must contain 18 training jobs, got {len(jobs)}")
    keys = {(job["regime"], job["model"], job["arm"], int(job["seed"])) for job in jobs}
    expected = {
        *(('complete', model, 'full_dapo', 20260609) for model in MODELS),
        *(("matched", model, arm, 20260610) for model in MODELS for arm in MATCHED_ARMS),
    }
    if keys != expected:
        raise ValueError(
            f"Stage 05 training matrix mismatch; missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )
    return jobs


def build_jobs(training_jobs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    evaluation_root = args.output_root / "stage05_legacy_taco_dapo_ablation" / "evaluation" / "canonical_11_20260719"
    jobs: list[dict[str, Any]] = []
    for model in ("Qwen2.5-7B-Instruct", "Qwen3-8B-Base"):
        jobs.append(
            {
                "kind": "base",
                "label": f"{model}__base",
                "model": model,
                "training_seed": None,
                "base_model_path": str(args.model_root / model),
                "selection_path": None,
                "output_dir": str(evaluation_root / "base" / f"{model}__base"),
            }
        )
    for training in training_jobs:
        regime = str(training["regime"])
        seed = int(training["seed"])
        model = str(training["model"])
        arm = str(training["arm"])
        run_dir = Path(str(training["output_dir"]))
        jobs.append(
            {
                "kind": regime,
                "label": f"{model}__{arm}",
                "model": model,
                "arm": arm,
                "training_seed": seed,
                "base_model_path": str(args.model_root / model),
                "training_run_dir": str(run_dir),
                "selection_path": str(run_dir / "selected_adapter.json"),
                "output_dir": str(evaluation_root / regime / f"train_seed_{seed}" / f"{model}__{arm}"),
            }
        )
    return jobs


def validate_metrics(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data.get("tasks_requested", [])) != set(TASK_N):
        raise ValueError(f"{path}: tasks_requested is not the canonical 11-task set")
    for task, expected_n in TASK_N.items():
        metric = data.get(SUITE[task], {}).get(task)
        if not isinstance(metric, dict) or metric.get("n") != expected_n:
            raise ValueError(f"{path}: {task} expected n={expected_n}, got {metric}")
    gsm = data["diagnostic"]["gsm8k"]
    if gsm.get("formal_result") is not True:
        raise ValueError(f"{path}: GSM8K is not marked as the full formal split")


def selected_adapter(job: dict[str, Any], selector: Path) -> str:
    selection = Path(str(job["selection_path"]))
    run_dir = Path(str(job["training_run_dir"]))
    if not selection.exists():
        subprocess.run(
            [
                sys.executable,
                str(selector),
                "--run-dir",
                str(run_dir),
                "--adapter-root",
                str(run_dir),
                "--output",
                str(selection),
            ],
            check=True,
        )
    payload = json.loads(selection.read_text(encoding="utf-8"))
    adapter = Path(str(payload.get("selected", {}).get("adapter", "")))
    if not ((adapter / "adapter_model.safetensors").is_file() or (adapter / "adapter_model.bin").is_file()):
        raise FileNotFoundError(f"selected adapter is missing: {adapter}")
    return str(adapter)


def run_job(job: dict[str, Any], args: argparse.Namespace, evaluator: Path, selector: Path) -> None:
    output_dir = Path(str(job["output_dir"]))
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        validate_metrics(metrics_path)
        return
    command = [
        sys.executable,
        str(evaluator),
        "--data-root",
        str(args.data_test_root),
        "--model-name",
        str(job["label"]),
        "--base-model-path",
        str(job["base_model_path"]),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.evaluation_seed),
        "--tasks",
        "scoreable",
        "--skip-healthbench-generation",
    ]
    if job["kind"] != "base":
        command.extend(["--adapter-path", selected_adapter(job, selector)])
    subprocess.run(command, check=True)
    validate_metrics(metrics_path)


def main() -> None:
    args = parse_args()
    stage_dir = Path(__file__).resolve().parent
    repo_root = stage_dir.parents[1]
    evaluator = repo_root / "evaluation" / "legacy_suite.py"
    selector = stage_dir / "select_best_adapter.py"
    jobs = build_jobs(read_training_jobs(args.training_manifest), args)
    manifest_path = args.manifest_output or (
        args.output_root / "stage05_legacy_taco_dapo_ablation" / "evaluation" / "evaluation_job_matrix.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluation_contract": "canonical_11_20260719",
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
    selected = jobs if args.all else ([jobs[args.job_index]] if args.job_index is not None else [])
    if args.validate_only:
        selected = jobs
    for job in selected:
        metrics = Path(str(job["output_dir"])) / "metrics.json"
        if args.validate_only:
            validate_metrics(metrics)
        else:
            run_job(job, args, evaluator, selector)
    print(json.dumps({"manifest": str(manifest_path), "jobs": len(jobs), "processed": len(selected)}))


if __name__ == "__main__":
    main()
