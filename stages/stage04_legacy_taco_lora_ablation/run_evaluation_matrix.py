#!/usr/bin/env python3
"""Consume a Stage 04 evaluation matrix through the shared legacy evaluator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVALUATOR = SCRIPT_DIR.parents[1] / "evaluation" / "legacy_suite.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="Matrix JSON path, or '-' for stdin.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--job-index", type=int)
    selector.add_argument("--job-id")
    selector.add_argument("--all", action="store_true")
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest(source: str) -> dict[str, Any]:
    payload = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    manifest = json.loads(payload)
    if manifest.get("tasks_alias") != "scoreable":
        raise ValueError("Stage 04 formal evaluation matrix must use --tasks scoreable")
    tasks = manifest.get("paper_tasks") or []
    if manifest.get("paper_task_count") != 11 or len(tasks) != 11:
        raise ValueError("Stage 04 formal evaluation matrix must contain the canonical 11 tasks")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("evaluation matrix has no jobs")
    job_ids = [job.get("job_id") for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("evaluation matrix contains duplicate job_id values")
    base_counts = Counter(job.get("model") for job in jobs if job.get("kind") == "base")
    expected_models = set(manifest.get("models") or [])
    if set(base_counts) != expected_models or any(count != 1 for count in base_counts.values()):
        raise ValueError(f"base jobs must occur exactly once per model: {dict(base_counts)}")
    for job in jobs:
        if job.get("tasks") != "scoreable" or job.get("limit") != 0:
            raise ValueError(f"non-formal task settings in job {job.get('job_id')}")
        if job.get("kind") == "base" and job.get("adapter_path") is not None:
            raise ValueError(f"base job unexpectedly has an adapter: {job.get('job_id')}")
        if job.get("kind") == "adapter" and not job.get("adapter_path"):
            raise ValueError(f"adapter job is missing adapter_path: {job.get('job_id')}")
    return manifest


def select_jobs(manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = manifest["jobs"]
    if args.all:
        return jobs
    if args.job_index is not None:
        if args.job_index < 0 or args.job_index >= len(jobs):
            raise IndexError(f"job index {args.job_index} outside [0, {len(jobs) - 1}]")
        return [jobs[args.job_index]]
    selected = [job for job in jobs if job.get("job_id") == args.job_id]
    if not selected:
        raise KeyError(f"job id not found: {args.job_id}")
    return selected


def build_command(job: dict[str, Any], args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(args.evaluator),
        "--data-root",
        str(job["data_test_root"]),
        "--model-name",
        str(job["job_id"]),
        "--base-model-path",
        str(job["base_model_path"]),
        "--output-dir",
        str(job["output_dir"]),
        "--seed",
        str(job["evaluation_seed"]),
        "--tasks",
        "scoreable",
        "--limit",
        "0",
        "--temperature",
        "0.2",
        "--top-p",
        "0.95",
        "--skip-healthbench-generation",
    ]
    if job.get("adapter_path"):
        command.extend(["--adapter-path", str(job["adapter_path"])])
    if args.resume:
        command.append("--resume")
    return command


def validate_external_paths(job: dict[str, Any], evaluator: Path) -> None:
    required = {
        "evaluator": evaluator,
        "data_test_root": Path(str(job["data_test_root"])),
        "base_model_path": Path(str(job["base_model_path"])),
    }
    if job.get("adapter_path"):
        required["adapter_path"] = Path(str(job["adapter_path"]))
    missing = {label: str(path) for label, path in required.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"required external paths are missing: {missing}")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.matrix)
    selected = select_jobs(manifest, args)
    commands = []
    for job in selected:
        command = build_command(job, args)
        commands.append({"job_id": job["job_id"], "command": command})
        if not args.dry_run:
            validate_external_paths(job, args.evaluator)
            subprocess.run(command, check=True)
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "selected_job_count": len(selected),
                "commands": commands,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
