#!/usr/bin/env python3
"""Build Stage 2 TACO+LeetCode SFT files for strict overlap ablations."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any


DATA_ROOT = Path(os.environ["CG_DATA_ROOT"]).expanduser() if os.environ.get("CG_DATA_ROOT") else None


if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

SEED = 20260603
TARGET_COUNT = 14043

CORE_CATEGORIES = [
    "math_number_theory",
    "data_structure",
    "dynamic_programming",
    "greedy_search",
    "implementation_simulation",
    "graph",
]
SMALL_CATEGORIES = [
    "string",
    "other_tagged",
    "sorting",
    "geometry",
    "bit_manipulation",
    "recursion_backtracking",
]
FINAL_CATEGORIES = CORE_CATEGORIES + ["other_algorithm"]
ARMS = ["full_sft", "full_sft_14043"] + [f"no_{category}" for category in FINAL_CATEGORIES]
REMOVE_SETS = {
    **{category: {category} for category in CORE_CATEGORIES},
    "other_algorithm": set(SMALL_CATEGORIES),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code-data-dir",
        type=Path,
        default=DATA_ROOT / "code_data/merged_taco_leetcode_by_primary" if DATA_ROOT else None,
        required=DATA_ROOT is None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "stage2_taco_sft_clean_prompt" if DATA_ROOT else None,
        required=DATA_ROOT is None,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_solution(row: dict[str, Any]) -> bool:
    return bool((row.get("canonical_solution") or "").strip())


def has_question(row: dict[str, Any]) -> bool:
    return bool((row.get("question") or row.get("name") or "").strip())


def sft_ok(row: dict[str, Any]) -> bool:
    return has_solution(row) and has_question(row)


def parse_input_output(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def overlap_categories(row: dict[str, Any]) -> set[str]:
    categories = row.get("overlap_categories") or []
    if categories:
        return {str(category) for category in categories}
    primary = row.get("primary_category")
    return {str(primary)} if primary else set()


def seven_bucket(row: dict[str, Any]) -> str:
    primary = str(row.get("primary_category") or "other_algorithm")
    if primary in SMALL_CATEGORIES:
        return "other_algorithm"
    if primary in FINAL_CATEGORIES or primary == "no_tags":
        return primary
    return "other_algorithm"


def taco_prompt(row: dict[str, Any]) -> str:
    io_spec = parse_input_output(row.get("input_output"))
    starter = row.get("starter_code") or ""
    fn_name = io_spec.get("fn_name")
    if fn_name:
        task = (
            f"Write a Python function named `{fn_name}` that solves the programming problem.\n"
            "Return only valid Python code, with no explanation."
        )
    else:
        task = (
            "Write a complete Python 3 program that solves the programming problem.\n"
            "Read input from standard input and write output to standard output.\n"
            "Return only valid Python code, with no explanation."
        )
    parts = [task, "", "Problem:", row.get("question") or ""]
    if starter.strip():
        parts += ["", "Starter code:", starter]
    parts += ["", "Python code:"]
    return "\n".join(parts)


def leetcode_prompt(row: dict[str, Any]) -> str:
    starter = row.get("starter_code") or ""
    task = (
        "Write Python code that solves the programming problem.\n"
        "Return only valid Python code, with no explanation."
    )
    parts = [task, "", "Problem:", row.get("question") or row.get("name") or ""]
    if starter.strip():
        parts += ["", "Starter code and helper definitions:", starter]
    parts += ["", "Python code:"]
    return "\n".join(parts)


def build_prompt(row: dict[str, Any]) -> str:
    if row.get("dataset") == "leetcode2k":
        return leetcode_prompt(row)
    return taco_prompt(row)


def to_sft_record(row: dict[str, Any], arm: str, source_bucket: str, removed_category: str | None) -> dict[str, Any]:
    solution = (row.get("canonical_solution") or "").strip()
    prompt = build_prompt(row)
    return {
        "id": row.get("id"),
        "dataset": row.get("dataset", "taco"),
        "arm": arm,
        "source_bucket": source_bucket,
        "removed_category": removed_category,
        "primary_category": row.get("primary_category"),
        "overlap_categories": row.get("overlap_categories"),
        "source": row.get("source"),
        "difficulty": row.get("difficulty"),
        "name": row.get("name"),
        "task_id": row.get("task_id"),
        "tags": row.get("tags"),
        "raw_tags": row.get("raw_tags"),
        "skill_types": row.get("skill_types"),
        "prompt": prompt,
        "answer": "\n" + solution + "\n",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "\n" + solution + "\n"},
        ],
    }


def sample_stratified(rows: list[dict[str, Any]], n: int, seed: int, label: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(rows) < n:
        raise ValueError(f"Not enough rows for {label}: need {n}, got {len(rows)}")
    strata: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        strata[seven_bucket(row)].append(row)
    total = len(rows)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for key, bucket_rows in strata.items():
        exact = len(bucket_rows) * n / total
        floor = int(exact)
        allocations[key] = floor
        remainders.append((exact - floor, key))
    remaining = n - sum(allocations.values())
    for _, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocations[key] < len(strata[key]):
            allocations[key] += 1
            remaining -= 1
    if remaining != 0:
        raise RuntimeError(f"Could not allocate exact sample size for {label}: remaining={remaining}")

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for key in sorted(strata):
        bucket_rows = strata[key][:]
        rng.shuffle(bucket_rows)
        sampled.extend(bucket_rows[: allocations[key]])
    rng.shuffle(sampled)
    if len(sampled) != n:
        raise RuntimeError(f"Sample size mismatch for {label}: expected={n}, got={len(sampled)}")
    return sampled, dict(sorted(allocations.items()))


def counter_dict(counter: collections.Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def summarize_pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "by_dataset": counter_dict(collections.Counter(row.get("dataset", "taco") for row in rows)),
        "by_primary_category": counter_dict(collections.Counter(row.get("primary_category") for row in rows)),
        "by_seven_bucket": counter_dict(collections.Counter(seven_bucket(row) for row in rows)),
        "by_difficulty": counter_dict(collections.Counter(str(row.get("difficulty") or "MISSING") for row in rows)),
    }


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    for path in sorted(args.code_data_dir.glob("*.jsonl")):
        raw_rows.extend(read_jsonl(path))
    clean_rows = [row for row in raw_rows if sft_ok(row)]
    rejected_rows = [row for row in raw_rows if not sft_ok(row)]

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "manifest_policy": "counts_schema_provenance_and_hashes_only",
        "contains_example_ids": False,
        "contains_examples": False,
        "seed": args.seed,
        "target_count": args.target_count,
        "source_dir": str(args.code_data_dir),
        "output_dir": str(out_dir),
        "definition": "TACO+LeetCode clean SFT rebuild. Clean rows require non-empty canonical_solution and question. Leave-one-out arms remove by strict overlap_categories.",
        "final_categories": FINAL_CATEGORIES,
        "small_categories_merged_into_other_algorithm": SMALL_CATEGORIES,
        "arms_order": ARMS,
        "record_schema_fields": [
            "id",
            "dataset",
            "arm",
            "source_bucket",
            "removed_category",
            "primary_category",
            "overlap_categories",
            "source",
            "difficulty",
            "name",
            "task_id",
            "tags",
            "raw_tags",
            "skill_types",
            "prompt",
            "answer",
            "messages",
        ],
        "raw_pool": summarize_pool(raw_rows),
        "clean_pool": summarize_pool(clean_rows),
        "rejected_pool": summarize_pool(rejected_rows),
        "arms": {},
    }

    for arm_index, arm in enumerate(ARMS):
        removed_category: str | None = None
        if arm == "full_sft":
            selected = clean_rows[:]
            allocation = counter_dict(collections.Counter(seven_bucket(row) for row in selected))
            pool_summary = summarize_pool(clean_rows)
        elif arm == "full_sft_14043":
            selected, allocation = sample_stratified(
                clean_rows,
                args.target_count,
                args.seed + 100 + arm_index,
                arm,
            )
            pool_summary = summarize_pool(clean_rows)
        else:
            removed_category = arm.removeprefix("no_")
            remove_set = REMOVE_SETS[removed_category]
            remaining_pool = [row for row in clean_rows if not (overlap_categories(row) & remove_set)]
            selected, allocation = sample_stratified(
                remaining_pool,
                args.target_count,
                args.seed + 100 + arm_index,
                arm,
            )
            pool_summary = summarize_pool(remaining_pool)

        sft_rows = [
            to_sft_record(row, arm, seven_bucket(row), removed_category)
            for row in selected
        ]
        path = out_dir / f"{arm}.jsonl"
        write_jsonl(path, sft_rows)
        manifest["arms"][arm] = {
            "train_file": str(path),
            "removed_category": removed_category,
            "remove_set": sorted(REMOVE_SETS[removed_category]) if removed_category else [],
            "pool_before_sampling": pool_summary,
            "sample_allocation_by_seven_bucket": allocation,
            "total": len(sft_rows),
            "empty_answer_after_filter": sum(1 for row in sft_rows if not (row.get("answer") or "").strip()),
            "by_dataset": counter_dict(collections.Counter(row.get("dataset", "taco") for row in sft_rows)),
            "by_difficulty": counter_dict(collections.Counter(str(row.get("difficulty") or "MISSING") for row in sft_rows)),
            "by_source_bucket": counter_dict(collections.Counter(row.get("source_bucket") for row in sft_rows)),
            "sha256": sha256_file(path),
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest={manifest_path}")
    print(f"raw_total={manifest['raw_pool']['total']}")
    print(f"clean_total={manifest['clean_pool']['total']}")
    print(f"rejected_total={manifest['rejected_pool']['total']}")
    for arm, item in manifest["arms"].items():
        print(f"{arm}: total={item['total']} by_dataset={item['by_dataset']} file={item['train_file']}")


if __name__ == "__main__":
    main()
