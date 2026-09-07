#!/usr/bin/env python3
"""Build equal-size Stage 3 DAPO ablation datasets from the verified full pool."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DATA_ROOT = Path(os.environ["CG_DATA_ROOT"]).expanduser() if os.environ.get("CG_DATA_ROOT") else None

FORMAL_CATEGORIES = {
    "no_tags",
    "data_structure",
    "math_number_theory",
    "dynamic_programming",
    "greedy_search",
    "implementation_simulation",
    "graph",
    "other_algorithm",
}

ABLATION_ARMS = [
    "full_dapo_12168",
    "no_math_number_theory",
    "no_data_structure",
    "no_dynamic_programming",
    "no_greedy_search",
    "no_implementation_simulation",
    "no_graph",
    "no_other_algorithm",
]

RAW_TO_FORMAL = {
    "no_tags": "no_tags",
    "data_structure": "data_structure",
    "math_number_theory": "math_number_theory",
    "dynamic_programming": "dynamic_programming",
    "greedy_search": "greedy_search",
    "implementation_simulation": "implementation_simulation",
    "graph": "graph",
    "sorting": "other_algorithm",
    "string": "other_algorithm",
    "bit_manipulation": "other_algorithm",
    "recursion_backtracking": "other_algorithm",
    "geometry": "other_algorithm",
    "other_tagged": "other_algorithm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-data-dir",
        type=Path,
        default=DATA_ROOT / "stage3_dapo_full_verified" if DATA_ROOT else None,
        required=DATA_ROOT is None,
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DATA_ROOT / "code_data/merged_taco_leetcode_train_index.jsonl" if DATA_ROOT else None,
        required=DATA_ROOT is None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATA_ROOT / "stage3_dapo_ablation_12168" if DATA_ROOT else None,
        required=DATA_ROOT is None,
    )
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--target-count", type=int, default=12168)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_seed(seed: int, arm: str) -> int:
    digest = hashlib.sha256(f"{seed}:{arm}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def map_category(category: str | None) -> str:
    if not category:
        return "no_tags"
    return RAW_TO_FORMAL.get(category, "other_algorithm")


def load_category_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            raw_overlap = item.get("overlap_categories") or []
            if not raw_overlap:
                raw_overlap = [item.get("primary_category")]
            overlap = sorted({map_category(category) for category in raw_overlap})
            primary = map_category(item.get("primary_category"))
            result[item["id"]] = {
                "primary_category": primary,
                "overlap_categories": overlap,
                "dataset": item.get("dataset"),
            }
    return result


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")


def clone_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned = [copy.deepcopy(record) for record in records]
    for index, record in enumerate(cloned):
        record.setdefault("extra_info", {})["index"] = index
    return cloned


def summarize(records: list[dict[str, Any]], category_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dataset_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    test_counts: list[int] = []
    prompt_lengths: list[int] = []
    missing: list[str] = []
    for record in records:
        info = record.get("extra_info") or {}
        problem_id = str(info.get("problem_id"))
        meta = category_index.get(problem_id)
        if meta is None:
            missing.append(problem_id)
            continue
        dataset_counts[str(info.get("dataset") or meta.get("dataset"))] += 1
        primary_counts[meta["primary_category"]] += 1
        for category in meta["overlap_categories"]:
            overlap_counts[category] += 1
        if "test_count_used" in info:
            test_counts.append(int(info["test_count_used"]))
        if "prompt_tokens_max" in info:
            prompt_lengths.append(int(info["prompt_tokens_max"]))
    return {
        "rows": len(records),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "primary_category_counts": {category: primary_counts.get(category, 0) for category in sorted(FORMAL_CATEGORIES)},
        "overlap_category_counts": {category: overlap_counts.get(category, 0) for category in sorted(FORMAL_CATEGORIES)},
        "test_count_min": min(test_counts) if test_counts else None,
        "test_count_max": max(test_counts) if test_counts else None,
        "test_count_mean": sum(test_counts) / len(test_counts) if test_counts else None,
        "prompt_tokens_max": max(prompt_lengths) if prompt_lengths else None,
        "missing_category_index": missing[:20],
        "missing_category_index_count": len(missing),
    }


def main() -> None:
    args = parse_args()
    seed_dir = args.output_root / f"seed_{args.seed}"
    if seed_dir.exists() and any(seed_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{seed_dir} already exists; pass --overwrite to rebuild")
    seed_dir.mkdir(parents=True, exist_ok=True)

    train_records = pq.read_table(args.full_data_dir / "train.parquet").to_pylist()
    val_records = pq.read_table(args.full_data_dir / "val.parquet").to_pylist()
    category_index = load_category_index(args.index_path)

    indexed_train: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    missing_train: list[str] = []
    for index, record in enumerate(train_records):
        problem_id = str(record.get("extra_info", {}).get("problem_id"))
        meta = category_index.get(problem_id)
        if meta is None:
            missing_train.append(problem_id)
            continue
        indexed_train.append((index, record, meta))
    if missing_train:
        raise SystemExit(f"Missing category index for {len(missing_train)} train rows; first={missing_train[:5]}")

    summary_rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for arm in ABLATION_ARMS:
        if arm == "full_dapo_12168":
            pool = indexed_train
            ablated_category = None
        else:
            ablated_category = arm.removeprefix("no_")
            pool = [
                item
                for item in indexed_train
                if ablated_category not in item[2]["overlap_categories"]
            ]
        if len(pool) < args.target_count:
            raise SystemExit(f"{arm} pool has only {len(pool)} rows, target={args.target_count}")

        rng = random.Random(stable_seed(args.seed, arm))
        shuffled = list(pool)
        rng.shuffle(shuffled)
        selected = shuffled[: args.target_count]
        selected_records = clone_records([item[1] for item in selected])
        selected_ids = [str(record["extra_info"]["problem_id"]) for record in selected_records]

        arm_dir = seed_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(arm_dir / "train.parquet", selected_records)
        write_parquet(arm_dir / "val.parquet", clone_records(val_records))

        with (arm_dir / "selected_problem_ids.txt").open("w", encoding="utf-8") as handle:
            for problem_id in selected_ids:
                handle.write(problem_id + "\n")

        removed_count = len(indexed_train) - len(pool)
        train_summary = summarize(selected_records, category_index)
        val_summary = summarize(val_records, category_index)
        manifest = {
            "seed": args.seed,
            "arm": arm,
            "target_count": args.target_count,
            "source_full_data_dir": str(args.full_data_dir),
            "source_index_path": str(args.index_path),
            "definition": (
                "full_dapo_12168 samples from the full verified train pool; "
                "no_* arms first remove any row whose formal overlap category contains "
                "the ablated category, then sample target_count rows deterministically."
            ),
            "sample_seed_material": f"{args.seed}:{arm}",
            "sample_seed_int": stable_seed(args.seed, arm),
            "ablated_category": ablated_category,
            "source_train_rows": len(indexed_train),
            "pool_rows_before_sampling": len(pool),
            "removed_rows_before_sampling": removed_count,
            "validation_policy": "Shared validation split copied from stage3_dapo_full_verified/val.parquet.",
            "train": train_summary,
            "validation": val_summary,
        }
        (arm_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifests[arm] = manifest
        summary_rows.append(
            {
                "seed": args.seed,
                "arm": arm,
                "train_rows": train_summary["rows"],
                "pool_rows_before_sampling": len(pool),
                "removed_rows_before_sampling": removed_count,
                "taco": train_summary["dataset_counts"].get("taco", 0),
                "leetcode2k": train_summary["dataset_counts"].get("leetcode2k", 0),
                **{
                    f"primary_{category}": train_summary["primary_category_counts"].get(category, 0)
                    for category in sorted(FORMAL_CATEGORIES)
                },
            }
        )

    with (seed_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (seed_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "target_count": args.target_count,
                "arms": ABLATION_ARMS,
                "arm_manifests": {
                    arm: f"{arm}/manifest.json" for arm in ABLATION_ARMS
                },
                "summary_csv": "summary.csv",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"seed_dir": str(seed_dir), "rows": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
