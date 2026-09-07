#!/usr/bin/env python3
"""Rebuild Stage-2 SFT data from execution-verified solutions.

The script never mutates the legacy dataset. It writes a versioned directory
containing a drop-in ``full_sft.jsonl``, a validation split, rejected records,
per-record audit data, and a manifest. Existing DAPO preflight results are used
as the trusted positive set. For normalized-but-unverified rows, a small set of
mechanical Python repairs can optionally be executed against all available
tests; no row is admitted unless every test passes.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


DATA_ROOT = Path(os.environ["CG_DATA_ROOT"]).expanduser() if os.environ.get("CG_DATA_ROOT") else None

from prepare_stage3_dapo_full_data import normalize_row  # noqa: E402
from stage3_code_reward import compute_score  # noqa: E402


DEFAULT_INPUT = DATA_ROOT / "stage2_taco_sft_clean_prompt/full_sft.jsonl" if DATA_ROOT else None
DEFAULT_RL_SOURCE = DATA_ROOT / "clean_data/merged_clean_rl_dedup.jsonl" if DATA_ROOT else None
DEFAULT_DAPO_DIR = DATA_ROOT / "stage3_dapo_full_verified" if DATA_ROOT else None
DEFAULT_OUTPUT = DATA_ROOT / "stage2_taco_sft_exec_verified_v2" if DATA_ROOT else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, required=DEFAULT_INPUT is None)
    parser.add_argument(
        "--rl-source", type=Path, default=DEFAULT_RL_SOURCE, required=DEFAULT_RL_SOURCE is None
    )
    parser.add_argument("--dapo-dir", type=Path, default=DEFAULT_DAPO_DIR, required=DEFAULT_DAPO_DIR is None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, required=DEFAULT_OUTPUT is None)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--repair-workers", type=int, default=24)
    parser.add_argument("--exec-timeout", type=float, default=180.0)
    parser.add_argument("--per-test-timeout", type=float, default=1.0)
    parser.add_argument("--attempt-simple-repairs", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalized_text(value: str) -> str:
    return " ".join(value.split())


def solution_text(row: dict[str, Any]) -> str:
    return str(row.get("canonical_solution") or "").strip()


def output_contract(meta: dict[str, Any], row: dict[str, Any], solution: str) -> str:
    kind = meta["interface_kind"]
    if kind == "taco_stdin":
        return "stdin_program"
    starter = str(row.get("starter_code") or "")
    if kind == "leetcode_function" or re.search(r"\bclass\s+Solution\b", starter + "\n" + solution):
        return "class_solution_method"
    return "standalone_function"


def render_prompt(row: dict[str, Any], meta: dict[str, Any], contract: str) -> str:
    question = str(row.get("question") or row.get("name") or "").strip()
    entry = str(meta.get("entry_point") or "").strip()
    starter = str(row.get("starter_code") or "").strip()
    if contract == "stdin_program":
        instruction = (
            "Write a complete Python 3 program that solves the programming problem.\n"
            "Read input from standard input and write only the required output to standard output.\n"
            "Return only executable Python code, with no explanation or Markdown fences."
        )
    elif contract == "class_solution_method":
        instruction = (
            "Write a complete Python file defining `class Solution` with a callable method "
            f"named `{entry}`.\nDo not read standard input or print diagnostics.\n"
            "Return only executable Python code, with no explanation or Markdown fences."
        )
    else:
        instruction = (
            f"Write a complete Python file defining a top-level function named `{entry}`.\n"
            "Do not read standard input or print diagnostics.\n"
            "Return only executable Python code, with no explanation or Markdown fences."
        )
    parts = [instruction, "", "Problem:", question]
    if starter:
        parts.extend(["", "Starter signature/imports:", "```python", starter, "```"])
    parts.extend(["", "Python code:"])
    return "\n".join(parts)


def to_sft_record(
    legacy: dict[str, Any],
    source: dict[str, Any],
    meta: dict[str, Any],
    solution: str,
    verification_source: str,
    repair_kind: str | None,
) -> dict[str, Any]:
    contract = output_contract(meta, source, solution)
    prompt = render_prompt(source, meta, contract)
    answer = "\n" + solution.strip() + "\n"
    return {
        "id": legacy["id"],
        "dataset": legacy.get("dataset"),
        "arm": "full_sft",
        "source_bucket": legacy.get("source_bucket"),
        "removed_category": None,
        "primary_category": legacy.get("primary_category"),
        "overlap_categories": legacy.get("overlap_categories"),
        "source": legacy.get("source"),
        "difficulty": legacy.get("difficulty"),
        "name": legacy.get("name"),
        "task_id": legacy.get("task_id"),
        "tags": legacy.get("tags"),
        "raw_tags": legacy.get("raw_tags"),
        "skill_types": legacy.get("skill_types"),
        "interface_family": "stdin" if meta["interface_kind"] == "taco_stdin" else "function",
        "interface_kind": meta["interface_kind"],
        "output_contract": contract,
        "entry_point": meta.get("entry_point"),
        "test_count": len(meta["tests"]),
        "verification": {
            "all_tests_passed": True,
            "source": verification_source,
            "repair_kind": repair_kind,
        },
        "prompt": prompt,
        "answer": answer,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def strip_markdown_fence(code: str) -> str:
    text = code.strip()
    match = re.fullmatch(r"```(?:python|py)?\s*\n(.*)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def append_main_guard(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    has_main_def = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main" for node in tree.body)
    if not has_main_def or re.search(r"\bmain\s*\(\s*\)", code):
        return None
    return code.rstrip() + "\n\nif __name__ == \"__main__\":\n    main()\n"


def python2_to_python3(code: str) -> str | None:
    try:
        from lib2to3.refactor import RefactoringTool, get_fixers_from_package

        tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
        return str(tool.refactor_string(code.rstrip() + "\n", "candidate")).strip()
    except Exception:
        return None


def repair_candidates(code: str) -> list[tuple[str, str]]:
    base = strip_markdown_fence(code).replace("\r\n", "\n")
    raw: list[tuple[str, str | None]] = []
    try:
        ast.parse(base)
        syntax_error = False
    except SyntaxError:
        syntax_error = True
    if base != code.strip():
        raw.append(("strip_markdown_fence", base))
    if syntax_error and "\t" in base:
        raw.append(("expand_tabs", base.expandtabs(4)))
    with_main = append_main_guard(base)
    if with_main:
        raw.append(("append_main", with_main))
    python2_pattern = re.compile(
        r"\b(raw_input|xrange)\s*\(|\.iter(items|keys|values)\s*\(|"
        r"(^|\n)\s*print\s+[^\(\n]|(^|\n)\s*except\s+[^:\n]+,\s*\w+\s*:",
        flags=re.MULTILINE,
    )
    converted = python2_to_python3(base) if python2_pattern.search(base) else None
    if converted and converted != base:
        raw.append(("python2_to_python3", converted))
        converted_with_main = append_main_guard(converted)
        if converted_with_main:
            raw.append(("python2_to_python3_and_main", converted_with_main))
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for name, candidate in raw:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result.append((name, candidate))
    return result


def verify_candidate(
    solution: str,
    meta: dict[str, Any],
    exec_timeout: float,
    per_test_timeout: float,
) -> dict[str, Any]:
    return compute_score(
        data_source=f"stage2_repair_{meta['dataset']}",
        solution_str=solution,
        ground_truth=meta,
        exec_timeout=exec_timeout,
        per_test_timeout=per_test_timeout,
    )


def load_dapo_records(dapo_dir: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    records: dict[str, dict[str, Any]] = {}
    val_ids: set[str] = set()
    for name in ["train.parquet", "val.parquet"]:
        table = pq.read_table(dapo_dir / name, columns=["reward_model", "extra_info"])
        for item in table.to_pylist():
            info = item["extra_info"]
            problem_id = info["problem_id"]
            if not info.get("canonical_verified"):
                continue
            records[problem_id] = json.loads(item["reward_model"]["ground_truth"])
            if name == "val.parquet":
                val_ids.add(problem_id)
    return records, val_ids


def stratified_validation(
    rows: list[dict[str, Any]],
    n: int,
    seed: int,
    forced_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n = min(n, max(0, len(rows) // 10))
    if n == 0:
        return rows, []
    forced = [row for row in rows if row["id"] in forced_ids]
    remaining_rows = [row for row in rows if row["id"] not in forced_ids]
    remaining_n = max(0, n - len(forced))
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in remaining_rows:
        buckets[(str(row.get("interface_kind")), str(row.get("source_bucket")))].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    exact = {
        key: remaining_n * len(bucket) / max(len(remaining_rows), 1)
        for key, bucket in buckets.items()
    }
    allocation = {key: int(value) for key, value in exact.items()}
    remaining = remaining_n - sum(allocation.values())
    for key in sorted(exact, key=lambda item: exact[item] - allocation[item], reverse=True)[:remaining]:
        allocation[key] += 1
    validation: list[dict[str, Any]] = forced[:]
    train: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        validation.extend(bucket[: allocation[key]])
        train.extend(bucket[allocation[key] :])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "by_dataset": dict(collections.Counter(str(row.get("dataset")) for row in rows)),
        "by_interface_kind": dict(collections.Counter(str(row.get("interface_kind")) for row in rows)),
        "by_output_contract": dict(collections.Counter(str(row.get("output_contract")) for row in rows)),
        "by_source_bucket": dict(collections.Counter(str(row.get("source_bucket")) for row in rows)),
        "tests_total": sum(int(row.get("test_count") or 0) for row in rows),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    legacy_rows = read_jsonl(args.input)
    legacy_by_id = {row["id"]: row for row in legacy_rows}
    source_by_id: dict[str, dict[str, Any]] = {}
    with args.rl_source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("id") in legacy_by_id:
                    source_by_id[row["id"]] = row

    verified_meta, dapo_val_ids = load_dapo_records(args.dapo_dir)
    accepted: list[dict[str, Any]] = []
    audit: dict[str, dict[str, Any]] = {}
    repair_jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    for problem_id, legacy in legacy_by_id.items():
        source = source_by_id.get(problem_id)
        if source is None:
            audit[problem_id] = {"id": problem_id, "status": "rejected", "reason": "not_in_rl_dedup_source"}
            continue
        meta, reason = normalize_row(source)
        if meta is None:
            audit[problem_id] = {"id": problem_id, "status": "rejected", "reason": reason}
            continue
        if problem_id in verified_meta:
            solution = solution_text(source)
            accepted.append(to_sft_record(legacy, source, verified_meta[problem_id], solution, "dapo_canonical_preflight", None))
            audit[problem_id] = {
                "id": problem_id,
                "status": "kept_verified",
                "reason": "canonical_full_pass",
                "interface_kind": meta["interface_kind"],
                "test_count": len(meta["tests"]),
                "was_dapo_validation": problem_id in dapo_val_ids,
            }
        elif args.attempt_simple_repairs:
            if repair_candidates(solution_text(source)):
                repair_jobs.append((problem_id, source, meta))
            else:
                audit[problem_id] = {
                    "id": problem_id,
                    "status": "rejected",
                    "reason": "canonical_failed_no_safe_mechanical_repair",
                    "interface_kind": meta["interface_kind"],
                    "test_count": len(meta["tests"]),
                }
        else:
            audit[problem_id] = {
                "id": problem_id,
                "status": "rejected",
                "reason": "canonical_not_execution_verified",
                "interface_kind": meta["interface_kind"],
                "test_count": len(meta["tests"]),
            }

    def repair_one(job: tuple[str, dict[str, Any], dict[str, Any]]) -> tuple[str, str | None, str | None, dict[str, Any] | None]:
        problem_id, source, meta = job
        last: dict[str, Any] | None = None
        for kind, candidate in repair_candidates(solution_text(source)):
            last = verify_candidate(candidate, meta, args.exec_timeout, args.per_test_timeout)
            if float(last.get("passed", 0.0)) == 1.0:
                return problem_id, kind, candidate, last
        return problem_id, None, None, last

    if repair_jobs:
        completed = 0
        with ThreadPoolExecutor(max_workers=args.repair_workers) as executor:
            futures = {executor.submit(repair_one, job): job for job in repair_jobs}
            for future in as_completed(futures):
                problem_id, kind, repaired_solution, result = future.result()
                source = source_by_id[problem_id]
                meta, _ = normalize_row(source)
                assert meta is not None
                if repaired_solution is not None:
                    accepted.append(
                        to_sft_record(
                            legacy_by_id[problem_id],
                            source,
                            meta,
                            repaired_solution,
                            "stage2_simple_repair_full_test",
                            kind,
                        )
                    )
                    audit[problem_id] = {
                        "id": problem_id,
                        "status": "repaired_verified",
                        "reason": kind,
                        "interface_kind": meta["interface_kind"],
                        "test_count": len(meta["tests"]),
                    }
                else:
                    audit[problem_id] = {
                        "id": problem_id,
                        "status": "rejected",
                        "reason": "all_simple_repairs_failed",
                        "interface_kind": meta["interface_kind"],
                        "test_count": len(meta["tests"]),
                        "last_status_code": None if result is None else result.get("status_code"),
                        "last_pass_frac": None if result is None else result.get("pass_frac"),
                    }
                completed += 1
                if completed % 100 == 0:
                    print(f"repair_progress={completed}/{len(repair_jobs)}", flush=True)

    accepted.sort(key=lambda row: row["id"])
    deduped: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    seen: set[str] = set()
    for row in accepted:
        key = hashlib.sha256(
            (normalized_text(row["prompt"]) + "\0" + normalized_text(row["answer"])).encode("utf-8")
        ).hexdigest()
        if key in seen:
            duplicate_ids.append(row["id"])
            audit[row["id"]]["status"] = "rejected"
            audit[row["id"]]["reason"] = "exact_prompt_answer_duplicate"
            continue
        seen.add(key)
        deduped.append(row)

    train, validation = stratified_validation(deduped, args.validation_count, args.seed, dapo_val_ids)
    rejected_rows = [
        {"audit": audit[row["id"]], "legacy_record": row}
        for row in legacy_rows
        if audit[row["id"]]["status"] == "rejected"
    ]
    audit_rows = [audit[row["id"]] for row in legacy_rows]

    write_jsonl(args.output_dir / "full_sft.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    write_jsonl(args.output_dir / "all_verified.jsonl", deduped)
    write_jsonl(args.output_dir / "rejected.jsonl", rejected_rows)
    write_jsonl(args.output_dir / "audit.jsonl", audit_rows)

    status_counts = dict(collections.Counter(row["status"] for row in audit_rows))
    rejection_reasons = dict(
        collections.Counter(row["reason"] for row in audit_rows if row["status"] == "rejected")
    )
    manifest = {
        "version": "stage2_taco_sft_exec_verified_v2",
        "seed": args.seed,
        "source": str(args.input),
        "rl_source": str(args.rl_source),
        "dapo_preflight_dir": str(args.dapo_dir),
        "output_dir": str(args.output_dir),
        "policy": {
            "admission": "all available verifier tests must pass",
            "legacy_overwritten": False,
            "prompt_contracts": ["stdin_program", "standalone_function", "class_solution_method"],
            "mechanical_repairs_attempted": bool(args.attempt_simple_repairs),
            "validation_count": len(validation),
            "dapo_validation_ids_forced_into_validation": len(
                {row["id"] for row in validation} & dapo_val_ids
            ),
            "deduplication": "exact normalized prompt+answer SHA256",
        },
        "input_rows": len(legacy_rows),
        "status_counts": status_counts,
        "rejection_reasons": rejection_reasons,
        "duplicates_removed": len(duplicate_ids),
        "all_verified": summarize(deduped),
        "train": summarize(train),
        "validation": summarize(validation),
        "files": {
            "train": str(args.output_dir / "full_sft.jsonl"),
            "validation": str(args.output_dir / "validation.jsonl"),
            "all_verified": str(args.output_dir / "all_verified.jsonl"),
            "rejected": str(args.output_dir / "rejected.jsonl"),
            "audit": str(args.output_dir / "audit.jsonl"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
