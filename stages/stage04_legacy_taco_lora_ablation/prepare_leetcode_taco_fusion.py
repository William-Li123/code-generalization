#!/usr/bin/env python3
"""Classify LeetCodeDataset tags with the TACO code_data taxonomy and merge indexes."""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from pathlib import Path
from typing import Any


DATA_ROOT = Path(os.environ["CG_DATA_ROOT"]).expanduser() if os.environ.get("CG_DATA_ROOT") else None


CATEGORY_PRIORITY = [
    "graph",
    "dynamic_programming",
    "data_structure",
    "math_number_theory",
    "greedy_search",
    "string",
    "geometry",
    "bit_manipulation",
    "recursion_backtracking",
    "sorting",
    "implementation_simulation",
]

# LeetCode tag -> (TACO primary taxonomy bucket, audit rationale)
LEETCODE_TAG_RULES: dict[str, tuple[str, str]] = {
    "Array": ("data_structure", "Array is a concrete container tag, aligned with TACO arrays/data structures."),
    "String": ("string", "String is the direct LeetCode counterpart of TACO strings/string algorithms."),
    "Hash Table": ("data_structure", "Hash table is a concrete data structure; TACO maps hash/hash table/hashing to data_structure."),
    "Dynamic Programming": ("dynamic_programming", "Direct dynamic programming tag."),
    "Math": ("math_number_theory", "Direct math tag."),
    "Sorting": ("sorting", "Direct sorting tag."),
    "Greedy": ("greedy_search", "Direct greedy tag, aligned with TACO greedy algorithms."),
    "Binary Search": ("greedy_search", "TACO places binary search under greedy/search style decision procedures."),
    "Depth-First Search": ("graph", "TACO maps DFS/depth-first search/graph traversal to graph."),
    "Matrix": ("math_number_theory", "TACO maps matrix/matrices to math_number_theory; kept for taxonomy compatibility."),
    "Bit Manipulation": ("bit_manipulation", "Direct bit manipulation tag."),
    "Breadth-First Search": ("graph", "TACO maps BFS/breadth-first search/graph traversal to graph."),
    "Two Pointers": ("greedy_search", "TACO maps two pointers to greedy_search."),
    "Tree": ("data_structure", "TACO maps tree/tree algorithms to data_structure unless the tag is explicitly graph-theoretic."),
    "Prefix Sum": ("data_structure", "TACO maps prefix sum to data_structure as an auxiliary array/index structure."),
    "Simulation": ("implementation_simulation", "Direct simulation tag."),
    "Heap (Priority Queue)": ("data_structure", "Heap/priority queue is a concrete data structure."),
    "Counting": ("implementation_simulation", "Counting is a basic frequency/counting implementation technique in this taxonomy."),
    "Graph": ("graph", "Direct graph tag."),
    "Stack": ("data_structure", "Stack is a concrete data structure."),
    "Binary Tree": ("data_structure", "Binary tree is a tree data structure."),
    "Sliding Window": ("greedy_search", "Sliding window is a two-pointer/search-window technique; grouped with greedy/search."),
    "Enumeration": ("greedy_search", "Enumeration corresponds to TACO complete search/brute force style tags."),
    "Backtracking": ("recursion_backtracking", "Direct backtracking tag."),
    "Union Find": ("graph", "TACO maps DSU/disjoint set union to graph because it mainly models connectivity."),
    "Number Theory": ("math_number_theory", "Direct number theory tag."),
    "Monotonic Stack": ("data_structure", "Monotonic stack is a specialized stack data structure."),
    "Segment Tree": ("data_structure", "Segment tree is a range-query data structure."),
    "Linked List": ("data_structure", "Linked list is a concrete data structure."),
    "Trie": ("data_structure", "Trie is a concrete data structure, even when used for strings."),
    "Combinatorics": ("math_number_theory", "TACO maps combinatorics to math_number_theory."),
    "Bitmask": ("dynamic_programming", "TACO maps bitmasks to dynamic_programming because they often encode subset DP states."),
    "Divide and Conquer": ("recursion_backtracking", "TACO maps divide and conquer with recursion/backtracking."),
    "Recursion": ("recursion_backtracking", "Direct recursion tag."),
    "Memoization": ("dynamic_programming", "TACO maps memoization to dynamic_programming."),
    "Geometry": ("geometry", "Direct geometry tag."),
    "Hash Function": ("data_structure", "Hash functions support hash-based data structures; TACO maps hash/hashing to data_structure."),
    "String Matching": ("string", "String matching is a string algorithm tag."),
    "Shortest Path": ("graph", "Shortest path is a graph algorithm tag."),
    "Topological Sort": ("graph", "Topological sorting is a graph/DAG algorithm tag."),
    "Binary Indexed Tree": ("data_structure", "Fenwick/BIT is a range-query data structure."),
    "Ordered Set": ("data_structure", "Ordered set is a concrete ordered data structure."),
    "Queue": ("data_structure", "Queue is a concrete data structure."),
    "Rolling Hash": ("string", "Rolling hash is primarily used here as a string matching technique."),
    "Binary Search Tree": ("data_structure", "BST is a tree data structure."),
    "Game Theory": ("math_number_theory", "TACO maps games/game theory to math_number_theory."),
    "Brainteaser": ("implementation_simulation", "Brainteaser is closest to TACO puzzles/logic/ad-hoc implementation tags."),
    "Monotonic Queue": ("data_structure", "Monotonic queue is a specialized queue data structure."),
    "Merge Sort": ("sorting", "Merge sort is a sorting algorithm."),
    "Counting Sort": ("sorting", "Counting sort is a sorting algorithm."),
    "Quickselect": ("sorting", "Quickselect is an order-statistic partitioning method closest to sorting in this taxonomy."),
    "Suffix Array": ("string", "Suffix array is a string index/algorithm."),
    "Line Sweep": ("geometry", "Line sweep is most often a computational-geometry/interval sweep technique."),
    "Probability and Statistics": ("math_number_theory", "Probability/statistics is mathematical."),
    "Bucket Sort": ("sorting", "Bucket sort is a sorting algorithm."),
    "Minimum Spanning Tree": ("graph", "MST is a graph algorithm."),
    "Radix Sort": ("sorting", "Radix sort is a sorting algorithm."),
    "Eulerian Circuit": ("graph", "Eulerian circuit is a graph/path algorithm."),
    "Strongly Connected Component": ("graph", "SCC is a directed graph decomposition algorithm."),
    "Interactive": ("implementation_simulation", "Interactive is an I/O/protocol constraint, closest to implementation_simulation."),
    "Biconnected Component": ("graph", "Biconnected components are graph decomposition concepts."),
    "Concurrency": ("implementation_simulation", "Concurrency is a runtime/control-flow implementation topic outside the algorithm buckets."),
    "Randomized": ("implementation_simulation", "Randomized is an implementation strategy not represented by a more specific TACO bucket."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leetcode-dir",
        type=Path,
        default=(DATA_ROOT / "raw_data/newfacade_LeetCodeDataset_34803eb64eab" if DATA_ROOT else None),
        required=DATA_ROOT is None,
    )
    parser.add_argument(
        "--code-data-dir",
        type=Path,
        default=DATA_ROOT / "code_data" if DATA_ROOT else None,
        required=DATA_ROOT is None,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_tag(tag: str) -> str:
    return tag.strip().lower()


def categories_for(tags: list[str]) -> list[str]:
    categories = {
        LEETCODE_TAG_RULES[tag][0]
        for tag in tags
        if tag in LEETCODE_TAG_RULES
    }
    return [category for category in CATEGORY_PRIORITY if category in categories]


def primary_category(categories: list[str], has_tags: bool) -> str:
    if categories:
        return categories[0]
    return "other_tagged" if has_tags else "no_tags"


def pct(count: int, total: int) -> str:
    return "0.00%" if total == 0 else f"{count / total * 100:.2f}%"


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def counter_dict(counter: collections.Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in counter.most_common()}


def leetcode_url(task_id: str | None) -> str | None:
    if not task_id:
        return None
    return f"https://leetcode.com/problems/{task_id}/"


def build_leetcode_records(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    unmapped: collections.Counter[str] = collections.Counter()

    for idx, row in enumerate(rows):
        meta = row.get("meta") or {}
        tags = [str(tag) for tag in meta.get("tags") or []]
        for tag in tags:
            if tag not in LEETCODE_TAG_RULES:
                unmapped[tag] += 1
        mapped_categories = categories_for(tags)
        primary = primary_category(mapped_categories, bool(tags))
        difficulty = str(meta.get("difficulty") or "UNKNOWN").upper()
        task_id = str(row.get("task_id") or f"row-{idx}")
        problem_id = f"leetcode_{split}_{idx:05d}"
        all_normalized_tags = [normalized_tag(tag) for tag in tags]
        record = {
            "id": problem_id,
            "dataset": "leetcode2k",
            "source_split": split,
            "source": "leetcode",
            "difficulty": difficulty,
            "name": meta.get("question_title"),
            "task_id": task_id,
            "url": leetcode_url(task_id),
            "raw_tags": tags,
            "tags": tags,
            "skill_types": tags,
            "all_normalized_tags": all_normalized_tags,
            "overlap_categories": mapped_categories,
            "primary_category": primary,
            "question": row.get("query"),
            "starter_code": row.get("prompt"),
            "input_output": row.get("input_output"),
            "canonical_solution": row.get("completion"),
            "num_solutions": 1 if row.get("completion") else 0,
            "entry_point": row.get("entry_point"),
            "test": row.get("test"),
            "response": row.get("response"),
            "metadata": {
                "question_id": meta.get("question_id"),
                "estimated_date": meta.get("estimated_date"),
                "lang_code_note": "meta.lang_code is a starter/signature snippet, not a language label",
                "lang_code": meta.get("lang_code"),
            },
        }
        records.append(record)
        index_rows.append(
            {
                "id": problem_id,
                "dataset": "leetcode2k",
                "primary_category": primary,
                "overlap_categories": mapped_categories,
                "source": "leetcode",
                "difficulty": difficulty,
                "tags": tags,
                "raw_tags": tags,
                "skill_types": tags,
                "name": meta.get("question_title"),
                "task_id": task_id,
            }
        )
    return records, index_rows, dict(unmapped)


def summarize_records(index_rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_counts: collections.Counter[str] = collections.Counter()
    overlap_counts: collections.Counter[str] = collections.Counter()
    difficulty_counts: collections.Counter[str] = collections.Counter()
    source_counts: collections.Counter[str] = collections.Counter()
    raw_tag_counts: collections.Counter[str] = collections.Counter()

    for row in index_rows:
        primary = str(row.get("primary_category") or "MISSING")
        primary_counts[primary] += 1
        categories = row.get("overlap_categories") or []
        if categories:
            for category in categories:
                overlap_counts[str(category)] += 1
        else:
            overlap_counts[primary] += 1
        difficulty_counts[str(row.get("difficulty") or "MISSING")] += 1
        source_counts[str(row.get("source") or "MISSING")] += 1
        for tag in row.get("raw_tags") or []:
            raw_tag_counts[str(tag)] += 1

    return {
        "total_train_problems": len(index_rows),
        "primary_category_counts": counter_dict(primary_counts),
        "overlap_category_counts": counter_dict(overlap_counts),
        "difficulty_counts": counter_dict(difficulty_counts),
        "source_counts": counter_dict(source_counts),
        "raw_tag_counts": counter_dict(raw_tag_counts),
    }


def read_taco_index(code_data_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(code_data_dir / "taco_train_index.jsonl")
    for row in rows:
        row.setdefault("dataset", "taco")
        row.setdefault("name", None)
        row.setdefault("task_id", row.get("id"))
    return rows


def copy_taco_records_to_merged(code_data_dir: Path, merged_primary_dir: Path, overwrite: bool) -> None:
    if overwrite and merged_primary_dir.exists():
        shutil.rmtree(merged_primary_dir)
    merged_primary_dir.mkdir(parents=True, exist_ok=True)
    for src_path in sorted((code_data_dir / "taco_by_primary").glob("*.jsonl")):
        category = src_path.stem
        out_rows: list[dict[str, Any]] = []
        for row in read_jsonl(src_path):
            row = dict(row)
            row["dataset"] = "taco"
            out_rows.append(row)
        write_jsonl(merged_primary_dir / f"{category}.jsonl", out_rows)


def append_rows_by_primary(rows: list[dict[str, Any]], primary_dir: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row.get("primary_category") or "other_tagged")].append(row)
    for category, category_rows in grouped.items():
        path = primary_dir / f"{category}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for row in category_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_stats_markdown(title: str, stats: dict[str, Any], output_note: str) -> str:
    total = stats["total_train_problems"]
    primary_rows = [
        [category, count, pct(count, total)]
        for category, count in stats["primary_category_counts"].items()
    ]
    overlap_rows = [
        [category, count, pct(count, total)]
        for category, count in stats["overlap_category_counts"].items()
    ]
    difficulty_rows = [[key, value] for key, value in stats["difficulty_counts"].items()]
    source_rows = [[key, value] for key, value in stats["source_counts"].items()]
    raw_tag_rows = [[key, value] for key, value in list(stats["raw_tag_counts"].items())[:80]]
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- train problems counted: {total}",
            f"- output: `{output_note}`",
            "- Primary category counts are mutually exclusive: each problem is assigned to exactly one bucket by the existing TACO priority order.",
            "- Overlap category counts are multi-label: a problem contributes to every mapped category implied by its tags.",
            "",
            "## Primary Category Counts",
            "",
            render_table(["primary_category", "problems", "pct"], primary_rows),
            "",
            "## Overlap Category Counts",
            "",
            render_table(["category", "problems", "pct"], overlap_rows),
            "",
            "## Difficulty Counts",
            "",
            render_table(["difficulty", "problems"], difficulty_rows),
            "",
            "## Source Counts",
            "",
            render_table(["source", "problems"], source_rows),
            "",
            "## Top Raw Tags",
            "",
            render_table(["raw_tag", "problems"], raw_tag_rows),
            "",
        ]
    )


def render_mapping_markdown(tag_counts: dict[str, int]) -> str:
    rows = []
    for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
        category, rationale = LEETCODE_TAG_RULES[tag]
        rows.append([tag, count, category, rationale])
    return "\n".join(
        [
            "# LeetCode Tag To TACO-Type Mapping",
            "",
            "This file maps `newfacade/LeetCodeDataset` tags to the existing TACO `data_train/code_data` primary category taxonomy.",
            "",
            render_table(["leetcode_tag", "train_count", "category", "why"], rows),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    code_data_dir = args.code_data_dir
    leetcode_dir = args.leetcode_dir

    train_rows = read_jsonl(leetcode_dir / "train.jsonl")
    test_rows = read_jsonl(leetcode_dir / "test.jsonl")
    train_records, train_index, unmapped_train = build_leetcode_records(train_rows, "train")
    _, test_index, unmapped_test = build_leetcode_records(test_rows, "test")
    if unmapped_train or unmapped_test:
        raise ValueError(f"Unmapped LeetCode tags: train={unmapped_train}, test={unmapped_test}")

    leetcode_primary_dir = code_data_dir / "leetcode_by_primary"
    if args.overwrite and leetcode_primary_dir.exists():
        shutil.rmtree(leetcode_primary_dir)
    leetcode_primary_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in train_records:
        grouped[str(record["primary_category"])].append(record)
    for category, rows in grouped.items():
        write_jsonl(leetcode_primary_dir / f"{category}.jsonl", rows)

    write_jsonl(code_data_dir / "leetcode_train_records.jsonl", train_records)
    write_jsonl(code_data_dir / "leetcode_train_index.jsonl", train_index)

    leetcode_stats = summarize_records(train_index)
    (code_data_dir / "leetcode_category_stats.json").write_text(
        json.dumps(leetcode_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (code_data_dir / "leetcode_category_stats.md").write_text(
        render_stats_markdown("LeetCodeDataset Train Code-Type Statistics", leetcode_stats, "data_train/code_data"),
        encoding="utf-8",
    )

    tag_counts: collections.Counter[str] = collections.Counter()
    for row in train_index:
        tag_counts.update(row.get("raw_tags") or [])
    (code_data_dir / "leetcode_tag_to_category.json").write_text(
        json.dumps(
            {
                tag: {
                    "category": LEETCODE_TAG_RULES[tag][0],
                    "train_count": int(count),
                    "rationale": LEETCODE_TAG_RULES[tag][1],
                }
                for tag, count in tag_counts.most_common()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (code_data_dir / "leetcode_tag_to_category.md").write_text(
        render_mapping_markdown(counter_dict(tag_counts)),
        encoding="utf-8",
    )

    taco_index = read_taco_index(code_data_dir)
    merged_index = taco_index + train_index
    write_jsonl(code_data_dir / "merged_taco_leetcode_train_index.jsonl", merged_index)

    merged_primary_dir = code_data_dir / "merged_taco_leetcode_by_primary"
    copy_taco_records_to_merged(code_data_dir, merged_primary_dir, args.overwrite)
    append_rows_by_primary(train_records, merged_primary_dir)

    merged_stats = summarize_records(merged_index)
    merged_stats["component_counts"] = {
        "taco_train": len(taco_index),
        "leetcode2k_train": len(train_index),
    }
    (code_data_dir / "merged_taco_leetcode_category_stats.json").write_text(
        json.dumps(merged_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (code_data_dir / "merged_taco_leetcode_category_stats.md").write_text(
        render_stats_markdown("Merged TACO + LeetCodeDataset Train Code-Type Statistics", merged_stats, "data_train/code_data"),
        encoding="utf-8",
    )

    all_tag_counts: collections.Counter[str] = collections.Counter()
    for row in train_index + test_index:
        all_tag_counts.update(row.get("raw_tags") or [])
    union_mapping_path = code_data_dir / "leetcode_tag_to_category_all_splits.json"
    union_mapping_path.write_text(
        json.dumps(
            {
                tag: {
                    "category": LEETCODE_TAG_RULES[tag][0],
                    "all_split_count": int(count),
                    "train_count": int(tag_counts.get(tag, 0)),
                    "rationale": LEETCODE_TAG_RULES[tag][1],
                }
                for tag, count in all_tag_counts.most_common()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"leetcode_train={len(train_index)}")
    print(f"merged_train={len(merged_index)}")
    print(f"leetcode_stats={code_data_dir / 'leetcode_category_stats.md'}")
    print(f"merged_stats={code_data_dir / 'merged_taco_leetcode_category_stats.md'}")
    print(f"mapping={code_data_dir / 'leetcode_tag_to_category.md'}")
    print(f"merged_by_primary={merged_primary_dir}")


if __name__ == "__main__":
    main()
