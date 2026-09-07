#!/usr/bin/env python3
"""Validate a Stage 05 manifest and launch selected jobs through run_job.sh."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "job_id",
    "regime",
    "model",
    "arm",
    "seed",
    "expected_train_rows",
    "train_batch_size",
    "total_steps",
    "model_path",
    "data_dir",
    "max_response_length",
    "micro_batch_size_per_gpu",
    "output_dir",
    "work_dir",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--job-id")
    selection.add_argument("--index", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("Stage 05 manifest schema_version must be 2")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Stage 05 manifest has no jobs")
    if payload.get("job_count") != len(jobs):
        raise ValueError("Stage 05 manifest job_count does not match jobs")
    seen: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise TypeError(f"job {index} is not an object")
        missing = sorted(REQUIRED_FIELDS - set(job))
        if missing:
            raise ValueError(f"job {index} is missing fields: {missing}")
        job_id = str(job["job_id"])
        if job_id in seen:
            raise ValueError(f"duplicate job_id: {job_id}")
        seen.add(job_id)
        regime = str(job["regime"])
        if regime not in {"complete", "matched"}:
            raise ValueError(f"invalid regime in {job_id}: {regime}")
        expected_seed = 20260609 if regime == "complete" else 20260610
        expected_rows = 17249 if regime == "complete" else 12168
        if int(job["seed"]) != expected_seed or int(job["expected_train_rows"]) != expected_rows:
            raise ValueError(f"archived seed/row contract mismatch in {job_id}")
        batch = int(job["train_batch_size"])
        if batch <= 0:
            raise ValueError(f"train_batch_size must be positive in {job_id}")
        expected_steps = (expected_rows + batch - 1) // batch
        if int(job["total_steps"]) != expected_steps:
            raise ValueError(f"total_steps mismatch in {job_id}: expected {expected_steps}")
        expected_response = 3072 if job["model"] == "Qwen3-8B-Base" else 2048
        if int(job["max_response_length"]) != expected_response:
            raise ValueError(f"max_response_length mismatch in {job_id}")
        expected_micro_batch = 4 if job["model"] == "Qwen3-8B-Base" else 8
        if int(job["micro_batch_size_per_gpu"]) != expected_micro_batch:
            raise ValueError(f"micro_batch_size_per_gpu mismatch in {job_id}")
    return jobs


def select_jobs(jobs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all:
        return jobs
    if args.index is not None:
        if args.index < 0 or args.index >= len(jobs):
            raise IndexError(f"--index must be in [0, {len(jobs) - 1}]")
        return [jobs[args.index]]
    matches = [job for job in jobs if job["job_id"] == args.job_id]
    if len(matches) != 1:
        raise KeyError(f"job_id not found: {args.job_id}")
    return matches


def job_environment(job: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    work_dir = Path(str(job["work_dir"]))
    env.update(
        {
            "RUN_NAME": str(job["job_id"]),
            "MODEL_NAME": str(job["model"]),
            "SOURCE_MODEL_PATH": str(job["model_path"]),
            "ARM": str(job["arm"]),
            "SEED": str(job["seed"]),
            "DATA_SEED": str(job["seed"]),
            "DATA_DIR": str(job["data_dir"]),
            "RUN_DIR": str(job["output_dir"]),
            "SYSTEM_LOG": str(work_dir / "train.system.log"),
            "EXPECTED_TRAIN_ROWS": str(job["expected_train_rows"]),
            "TOTAL_STEPS": str(job["total_steps"]),
            "TRAIN_BATCH_SIZE": str(job["train_batch_size"]),
            "GEN_BATCH_SIZE": str(job["train_batch_size"]),
            "MAX_RESPONSE_LENGTH": str(job["max_response_length"]),
            "PPO_MICRO_BATCH_SIZE_PER_GPU": str(job["micro_batch_size_per_gpu"]),
            "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": str(job["micro_batch_size_per_gpu"]),
        }
    )
    return env


def main() -> None:
    args = parse_args()
    jobs = select_jobs(load_jobs(args.manifest), args)
    runner = Path(__file__).with_name("run_job.sh")
    for job in jobs:
        summary = {
            "job_id": job["job_id"],
            "runner": str(runner),
            "data_dir": job["data_dir"],
            "output_dir": job["output_dir"],
            "total_steps": job["total_steps"],
        }
        print(json.dumps(summary, ensure_ascii=False))
        if not args.dry_run:
            subprocess.run(["bash", str(runner)], env=job_environment(job), check=True)


if __name__ == "__main__":
    main()
