#!/usr/bin/env python3
"""Prepare TACO train split for code-type ablation experiments."""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DATA_ROOT = Path(os.environ["CG_DATA_ROOT"]).expanduser() if os.environ.get("CG_DATA_ROOT") else None


TAG_TO_CATEGORY = {
    # Dynamic programming
    "dynamic programming": "dynamic_programming",
    "dp": "dynamic_programming",
    "dynamicprogramming": "dynamic_programming",
    "algorithms - dynamic programming": "dynamic_programming",
    "memoization": "dynamic_programming",
    "bitmasks": "dynamic_programming",
    # Graphs
    "graph algorithms": "graph",
    "graph algos": "graph",
    "graph traversal": "graph",
    "graphs": "graph",
    "graph": "graph",
    "dfs and similar": "graph",
    "dfs": "graph",
    "bfs": "graph",
    "traversals": "graph",
    "depth-first search": "graph",
    "breadth-first search": "graph",
    "shortest paths": "graph",
    "spanning trees": "graph",
    "dsu": "graph",
    "disjoint set union": "graph",
    "2-sat": "graph",
    "graph matchings": "graph",
    "flows": "graph",
    # Data structures
    "data structures": "data_structure",
    "datastructures": "data_structure",
    "arrays": "data_structure",
    "array": "data_structure",
    "lists": "data_structure",
    "prefix sum": "data_structure",
    "trees": "data_structure",
    "tree": "data_structure",
    "tree algorithms": "data_structure",
    "hashing": "data_structure",
    "hash": "data_structure",
    "hash table": "data_structure",
    "heaps": "data_structure",
    "heap": "data_structure",
    "stacks": "data_structure",
    "stack": "data_structure",
    "queues": "data_structure",
    "queue": "data_structure",
    "linked lists": "data_structure",
    "linked list": "data_structure",
    "range queries": "data_structure",
    "segment tree": "data_structure",
    "binary indexed tree": "data_structure",
    "fenwick tree": "data_structure",
    # Mathematics
    "mathematics": "math_number_theory",
    "mathematical": "math_number_theory",
    "math": "math_number_theory",
    "basic maths": "math_number_theory",
    "arithmetic": "math_number_theory",
    "number theory": "math_number_theory",
    "mathematics - number theory": "math_number_theory",
    "modular arithmetic": "math_number_theory",
    "combinatorics": "math_number_theory",
    "probabilities": "math_number_theory",
    "probability": "math_number_theory",
    "matrices": "math_number_theory",
    "matrix": "math_number_theory",
    "fft": "math_number_theory",
    "chinese remainder theorem": "math_number_theory",
    "games": "math_number_theory",
    "game theory": "math_number_theory",
    # Greedy/search
    "greedy algorithms": "greedy_search",
    "greedy": "greedy_search",
    "searching": "greedy_search",
    "complete search": "greedy_search",
    "brute force": "greedy_search",
    "binary search": "greedy_search",
    "two pointers": "greedy_search",
    "ternary search": "greedy_search",
    "meet-in-the-middle": "greedy_search",
    "constructive algorithms": "greedy_search",
    "constructive": "greedy_search",
    # Sorting
    "sorting": "sorting",
    "sortings": "sorting",
    "amortized analysis": "sorting",
    # Strings
    "strings": "string",
    "string": "string",
    "string algorithms": "string",
    "string suffix structures": "string",
    "expression parsing": "string",
    "regular expressions": "string",
    # Geometry
    "geometry": "geometry",
    "computational geometry": "geometry",
    # Bit manipulation
    "bit manipulation": "bit_manipulation",
    "bits": "bit_manipulation",
    "bit magic": "bit_manipulation",
    "bitwise operation": "bit_manipulation",
    # Recursion/backtracking
    "recursion": "recursion_backtracking",
    "backtracking": "recursion_backtracking",
    "divide and conquer": "recursion_backtracking",
    # Implementation/simulation
    "implementation": "implementation_simulation",
    "simulation": "implementation_simulation",
    "fundamentals": "implementation_simulation",
    "algorithms": "implementation_simulation",
    "advanced algorithms": "implementation_simulation",
    "simple algos": "implementation_simulation",
    "basic programming concepts": "implementation_simulation",
    "basicprogramming": "implementation_simulation",
    "ad-hoc": "implementation_simulation",
    "observation": "implementation_simulation",
    "puzzles": "implementation_simulation",
    "case work": "implementation_simulation",
    "logic": "implementation_simulation",
    "conditional statements": "implementation_simulation",
}

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

TRAIN_COLUMNS = [
    "question",
    "solutions",
    "starter_code",
    "input_output",
    "difficulty",
    "raw_tags",
    "name",
    "source",
    "tags",
    "skill_types",
    "url",
    "time_limit",
    "memory_limit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taco-dir",
        type=Path,
        default=DATA_ROOT / "raw_data/TACO" if DATA_ROOT else None,
        required=DATA_ROOT is None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "code_data" if DATA_ROOT else None,
        required=DATA_ROOT is None,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    return parsed if isinstance(parsed, list) else []


def normalized_tags(*values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        for item in parse_list(value):
            text = str(item).strip()
            norm = text.lower()
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
    return out


def categories_for(tags: list[str]) -> list[str]:
    categories = {TAG_TO_CATEGORY[tag] for tag in tags if tag in TAG_TO_CATEGORY}
    return [category for category in CATEGORY_PRIORITY if category in categories]


def primary_category(categories: list[str], has_tags: bool) -> str:
    if categories:
        return categories[0]
    return "other_tagged" if has_tags else "no_tags"


def first_solution(value: Any) -> tuple[str, int]:
    solutions = parse_list(value)
    if not solutions:
        return "", 0
    return str(solutions[0]), len(solutions)


def write_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def pct(count: int, total: int) -> str:
    return "0.00%" if total == 0 else f"{count / total * 100:.2f}%"


def counter_dict(counter: collections.Counter) -> dict[str, int]:
    return {key: int(value) for key, value in counter.most_common()}


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(stats: dict[str, Any]) -> str:
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
    tag_rows = [[key, value] for key, value in list(stats["tag_counts"].items())[:80]]
    skill_rows = [[key, value] for key, value in list(stats["skill_type_counts"].items())[:80]]
    lines = [
        "# TACO Train Code-Type Statistics",
        "",
        f"- train problems counted: {total}",
        "- test split: excluded from these statistics",
        f"- output_dir: `{stats['output_dir']}`",
        "",
        "## Primary Category Counts",
        "",
        render_table(["primary_category", "problems", "pct_of_train"], primary_rows),
        "",
        "## Overlap Category Counts",
        "",
        render_table(["category", "problems", "pct_of_train"], overlap_rows),
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
        "## Top Normalized Tags",
        "",
        render_table(["tag", "problems"], tag_rows),
        "",
        "## Top Skill Types",
        "",
        render_table(["skill_type", "problems"], skill_rows),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    train_files = sorted((args.taco_dir / "ALL").glob("train-*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No train parquet files found under {args.taco_dir / 'ALL'}")

    output_dir = args.output_dir
    primary_dir = output_dir / "taco_by_primary"
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    primary_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "taco_dir": str(args.taco_dir),
        "output_dir": str(output_dir),
        "train_files": [str(path) for path in train_files],
        "test_excluded": True,
        "total_train_problems": 0,
        "primary_category_counts": collections.Counter(),
        "overlap_category_counts": collections.Counter(),
        "difficulty_counts": collections.Counter(),
        "source_counts": collections.Counter(),
        "raw_tag_counts": collections.Counter(),
        "tag_counts": collections.Counter(),
        "skill_type_counts": collections.Counter(),
        "no_tag_examples": [],
        "other_tagged_examples": [],
    }

    handles: dict[str, Any] = {}
    index_path = output_dir / "taco_train_index.jsonl"
    index_handle = index_path.open("w", encoding="utf-8")
    try:
        global_index = 0
        for path in train_files:
            table = pq.read_table(path, columns=TRAIN_COLUMNS)
            for row in table.to_pylist():
                raw_tags = [str(item) for item in parse_list(row.get("raw_tags"))]
                tags = [str(item) for item in parse_list(row.get("tags"))]
                skill_types = [str(item) for item in parse_list(row.get("skill_types"))]
                all_tags = normalized_tags(row.get("raw_tags"), row.get("tags"), row.get("skill_types"))
                categories = categories_for(all_tags)
                primary = primary_category(categories, bool(all_tags))
                solution, solution_count = first_solution(row.get("solutions"))
                problem_id = f"taco_train_{global_index:05d}"

                record = {
                    "id": problem_id,
                    "source_split": "train",
                    "source": row.get("source"),
                    "difficulty": row.get("difficulty"),
                    "name": row.get("name"),
                    "url": row.get("url"),
                    "raw_tags": raw_tags,
                    "tags": tags,
                    "skill_types": skill_types,
                    "all_normalized_tags": all_tags,
                    "overlap_categories": categories,
                    "primary_category": primary,
                    "question": row.get("question"),
                    "starter_code": row.get("starter_code"),
                    "input_output": row.get("input_output"),
                    "canonical_solution": solution,
                    "num_solutions": solution_count,
                    "time_limit": row.get("time_limit"),
                    "memory_limit": row.get("memory_limit"),
                }

                handle = handles.get(primary)
                if handle is None:
                    handle = (primary_dir / f"{primary}.jsonl").open("w", encoding="utf-8")
                    handles[primary] = handle
                write_jsonl(handle, record)
                write_jsonl(index_handle, {
                    "id": problem_id,
                    "primary_category": primary,
                    "overlap_categories": categories,
                    "source": row.get("source"),
                    "difficulty": row.get("difficulty"),
                    "tags": tags,
                    "raw_tags": raw_tags,
                    "skill_types": skill_types,
                })

                stats["total_train_problems"] += 1
                stats["primary_category_counts"][primary] += 1
                if categories:
                    for category in categories:
                        stats["overlap_category_counts"][category] += 1
                else:
                    stats["overlap_category_counts"][primary] += 1
                stats["difficulty_counts"][str(row.get("difficulty") or "MISSING")] += 1
                stats["source_counts"][str(row.get("source") or "MISSING")] += 1
                for tag in raw_tags:
                    stats["raw_tag_counts"][tag.lower()] += 1
                for tag in tags:
                    stats["tag_counts"][tag.lower()] += 1
                for tag in skill_types:
                    stats["skill_type_counts"][tag.lower()] += 1
                if primary == "no_tags" and len(stats["no_tag_examples"]) < 20:
                    stats["no_tag_examples"].append({"id": problem_id, "source": row.get("source"), "difficulty": row.get("difficulty")})
                if primary == "other_tagged" and len(stats["other_tagged_examples"]) < 20:
                    stats["other_tagged_examples"].append({"id": problem_id, "source": row.get("source"), "difficulty": row.get("difficulty"), "tags": tags, "raw_tags": raw_tags})
                global_index += 1
    finally:
        index_handle.close()
        for handle in handles.values():
            handle.close()

    for key in [
        "primary_category_counts",
        "overlap_category_counts",
        "difficulty_counts",
        "source_counts",
        "raw_tag_counts",
        "tag_counts",
        "skill_type_counts",
    ]:
        stats[key] = counter_dict(stats[key])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "taco_train_category_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "taco_train_category_stats.md").write_text(
        render_markdown(stats),
        encoding="utf-8",
    )

    print(f"total_train_problems={stats['total_train_problems']}")
    print(f"stats_json={output_dir / 'taco_train_category_stats.json'}")
    print(f"stats_md={output_dir / 'taco_train_category_stats.md'}")
    print(f"primary_dir={primary_dir}")


if __name__ == "__main__":
    main()
