#!/usr/bin/env python3
"""Re-score stored Gemma2 native-final MBPP+ generations.

The original evaluation process failed before executing candidate code because
its child process could not load libcblas.so.3. This script reuses the stored
generations and only repeats the deterministic MBPP+ test execution.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAIN_TASKS = [
    "arc_challenge",
    "legalbench",
    "math500_medium",
    "finqa",
    "medcalc",
    "humaneval",
    "mbpp_plus",
]
HARD_TASKS = ["math500_high_level"]
DIAGNOSTIC_TASKS = ["scienceqa", "gsm8k", "mbpp_simple"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def task_score(record: dict[str, Any]) -> float:
    for key in ("accuracy", "pass_at_1", "score"):
        if key in record:
            return float(record[key])
    raise KeyError(f"metric record has no score: {record}")


def recompute_means(metrics: dict[str, Any]) -> None:
    metrics["main"]["main_scoreable_mean"] = sum(
        task_score(metrics["main"][task]) for task in MAIN_TASKS
    ) / len(MAIN_TASKS)
    metrics["hard"]["hard_mean"] = sum(
        task_score(metrics["hard"][task]) for task in HARD_TASKS
    ) / len(HARD_TASKS)
    metrics["diagnostic"]["diagnostic_mean"] = sum(
        task_score(metrics["diagnostic"][task]) for task in DIAGNOSTIC_TASKS
    ) / len(DIAGNOSTIC_TASKS)
    all_scores = [
        *(task_score(metrics["main"][task]) for task in MAIN_TASKS),
        *(task_score(metrics["hard"][task]) for task in HARD_TASKS),
        *(task_score(metrics["diagnostic"][task]) for task in DIAGNOSTIC_TASKS),
    ]
    metrics["scoreable_overall_mean"] = sum(all_scores) / len(all_scores)


def run_candidate(
    executable: str,
    code: str,
    row: dict[str, Any],
    timeout_seconds: int,
) -> tuple[bool, str]:
    imports = "\n".join(row.get("test_imports") or [])
    tests = "\n".join(row.get("test_list") or [])
    plus_test = row.get("plus_test") or ""
    script = imports + "\n\n" + code + "\n\n" + tests + "\n" + plus_test + "\n"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", encoding="utf-8", delete=False
    ) as handle:
        handle.write(script)
        script_path = Path(handle.name)
    try:
        result = subprocess.run(
            [executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        script_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=12)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()

    old_records = read_jsonl(args.results_jsonl)
    dataset = {str(row["id"]): row for row in read_jsonl(args.dataset_jsonl)}
    missing = [
        str(row.get("id"))
        for row in old_records
        if str(row.get("id")) not in dataset
    ]
    if missing:
        raise RuntimeError(f"missing dataset rows: {missing[:10]}")

    results_backup = args.results_jsonl.with_suffix(".pre_repair.jsonl")
    metrics_backup = args.metrics_json.with_suffix(".pre_repair.json")
    if not results_backup.exists():
        shutil.copy2(args.results_jsonl, results_backup)
    if not metrics_backup.exists():
        shutil.copy2(args.metrics_json, metrics_backup)

    def check(record: dict[str, Any]) -> dict[str, Any]:
        updated = dict(record)
        code = str(record.get("extracted_code") or "")
        try:
            ast.parse(code)
            ok, output = run_candidate(
                args.python_executable,
                code,
                dataset[str(record["id"])],
                args.timeout_seconds,
            )
            status = (
                "passed"
                if ok
                else ("timeout" if output == "timeout" else "test_or_runtime_failure")
            )
        except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
            ok = False
            output = f"{type(exc).__name__}: {exc}"
            status = "code_parse_failure"
        updated["passed"] = bool(ok)
        updated["status"] = status
        updated["error"] = "" if ok else output[:4000]
        updated["repair"] = "libcblas_environment_rescore"
        return updated

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        new_records = list(executor.map(check, old_records))

    correct = sum(int(row["passed"]) for row in new_records)
    if len(new_records) != 378:
        raise RuntimeError(f"unexpected MBPP+ row count: {len(new_records)}")
    if all("libcblas.so.3" in str(row.get("error") or "") for row in new_records):
        raise RuntimeError("repair environment still has the original libcblas failure")

    write_jsonl(args.results_jsonl, new_records)
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    before = dict(metrics["main"]["mbpp_plus"])
    metrics["main"]["mbpp_plus"] = {
        "n": len(new_records),
        "correct": correct,
        "pass_at_1": correct / len(new_records),
    }
    recompute_means(metrics)
    metrics["mbpp_plus_repair"] = {
        "reason": "original child process could not load libcblas.so.3",
        "method": "re-score stored generations without regeneration",
        "repaired_at": datetime.now(timezone.utc).isoformat(),
    }
    args.metrics_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    status_counts: dict[str, int] = {}
    for record in new_records:
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    audit = {
        "rows": len(new_records),
        "before": before,
        "after": metrics["main"]["mbpp_plus"],
        "status_counts": status_counts,
        "results_backup": str(results_backup),
        "metrics_backup": str(metrics_backup),
        "python_executable": args.python_executable,
        "generation_reused": True,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
