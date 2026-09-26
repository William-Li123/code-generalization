#!/usr/bin/env python3
"""Build the full verified Stage 3 RLVR dataset with all available tests."""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from stage3_code_reward import compute_score


DATA_ROOT = Path(os.environ["CG_DATA_ROOT"]).expanduser() if os.environ.get("CG_DATA_ROOT") else None
MODEL_ROOT = Path(os.environ["CG_MODEL_ROOT"]).expanduser() if os.environ.get("CG_MODEL_ROOT") else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_ROOT / "clean_data/merged_clean_rl_dedup.jsonl" if DATA_ROOT else None,
        required=DATA_ROOT is None,
    )
    parser.add_argument(
        "--model-paths",
        type=Path,
        nargs="+",
        default=(
            [MODEL_ROOT / "Qwen3-8B-Base", MODEL_ROOT / "Qwen2.5-7B-Instruct"]
            if MODEL_ROOT
            else None
        ),
        required=MODEL_ROOT is None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "stage3_dapo_full_verified" if DATA_ROOT else None,
        required=DATA_ROOT is None,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--verify-workers", type=int, default=48)
    parser.add_argument("--exec-timeout", type=float, default=180.0)
    parser.add_argument("--per-test-timeout", type=float, default=1.0)
    return parser.parse_args()


def safe_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "None":
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def parse_call_kwargs(text: str) -> dict[str, Any]:
    expr = ast.parse(f"_f({text})", mode="eval").body
    if not isinstance(expr, ast.Call):
        raise ValueError("not a function call")
    kwargs: dict[str, Any] = {}
    for kw in expr.keywords:
        if kw.arg is None:
            raise ValueError("starred kwargs are unsupported")
        kwargs[kw.arg] = ast.literal_eval(kw.value)
    return kwargs


def load_io(row: dict[str, Any]) -> Any:
    raw = row.get("input_output")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def normalize_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = load_io(row)
    except Exception:
        return None, "invalid_input_output_json"

    dataset = row.get("dataset")
    mode = row.get("mode")
    tests: list[dict[str, Any]] = []
    entry_point = row.get("entry_point")
    interface_kind = ""

    if dataset == "leetcode2k":
        if mode != "function" or not entry_point or not isinstance(raw, list):
            return None, "leetcode_unsupported_interface"
        for item in raw:
            try:
                kwargs = parse_call_kwargs(item.get("input", ""))
            except Exception:
                return None, "leetcode_unparseable_call"
            tests.append({"kwargs": kwargs, "output": safe_literal(item.get("output"))})
        interface_kind = "leetcode_function"
    elif dataset == "taco" and mode == "stdin":
        if (row.get("starter_code") or "").strip():
            return None, "taco_pseudo_stdin_nonempty_starter"
        if not isinstance(raw, dict):
            return None, "taco_stdin_bad_io_type"
        inputs = raw.get("inputs") or []
        outputs = raw.get("outputs") or []
        if not inputs or len(inputs) != len(outputs):
            return None, "taco_stdin_unaligned_io"
        tests = [{"input": inp, "output": out} for inp, out in zip(inputs, outputs, strict=True)]
        entry_point = None
        interface_kind = "taco_stdin"
    elif dataset == "taco" and mode == "function":
        if not isinstance(raw, dict):
            return None, "taco_function_bad_io_type"
        fn_name = raw.get("fn_name") or entry_point
        inputs = raw.get("inputs") or []
        outputs = raw.get("outputs") or []
        if not fn_name or not inputs or len(inputs) != len(outputs):
            return None, "taco_function_unaligned_io"
        for inp, out in zip(inputs, outputs, strict=True):
            if not isinstance(inp, (list, tuple)):
                return None, "taco_function_nonpositional_input"
            expected = out[0] if isinstance(out, list) and len(out) == 1 else out
            tests.append({"args": list(inp), "output": expected})
        entry_point = str(fn_name)
        interface_kind = "taco_function"
    else:
        return None, "unsupported_dataset_or_mode"

    if not tests:
        return None, "no_normalized_tests"

    normalized = {
        "problem_id": row["id"],
        "dataset": dataset,
        "difficulty": row.get("difficulty"),
        "mode": mode,
        "interface_kind": interface_kind,
        "entry_point": entry_point,
        "starter_code": row.get("starter_code") or "",
        "tests": tests,
    }
    return normalized, None


def render_messages(row: dict[str, Any], meta: dict[str, Any]) -> list[dict[str, str]]:
    if meta["interface_kind"] == "leetcode_function":
        interface = f"""Interface contract:
- Return a complete Python file that defines `class Solution`.
- The grader executes the file and evaluates `{meta["entry_point"]}`.
- Do not read stdin or print diagnostics."""
    elif meta["interface_kind"] == "taco_function":
        interface = f"""Interface contract:
- Return a complete Python file defining callable `{meta["entry_point"]}`.
- The grader imports the file and calls that function with positional arguments.
- Do not read stdin or print diagnostics."""
    else:
        interface = """Interface contract:
- Return a complete standard-input Python 3 program.
- Read from stdin and write only the required answer to stdout.
- Handle multiple test cases when required by the statement."""

    starter = meta["starter_code"].strip()
    starter_block = f"\n\nStarter signature/imports:\n```python\n{starter}\n```" if starter else ""
    user = f"""Solve the following coding problem.

Output only executable Python code. Markdown fences and short surrounding text
are tolerated by the evaluator, but do not include local tests or diagnostics.

{interface}

Dataset: {meta["dataset"]}
Difficulty: {meta.get("difficulty")}
Problem id: {meta["problem_id"]}
Verifier test count: {len(meta["tests"])}

Problem statement:
{row.get("question") or ""}
{starter_block}
"""
    return [
        {
            "role": "system",
            "content": "You are a precise Python programming-problem solver.",
        },
        {"role": "user", "content": user},
    ]


def canonical_preflight(item: dict[str, Any], exec_timeout: float, per_test_timeout: float) -> tuple[str, bool, str]:
    solution = (item["row"].get("canonical_solution") or "").strip()
    if not solution:
        return item["meta"]["problem_id"], True, "no_canonical_structural_only"
    result = compute_score(
        data_source=f"stage3_{item['meta']['dataset']}",
        solution_str=solution,
        ground_truth=item["meta"],
        exec_timeout=exec_timeout,
        per_test_timeout=per_test_timeout,
    )
    if result["passed"] == 1.0:
        return item["meta"]["problem_id"], True, "canonical_full_pass"
    return item["meta"]["problem_id"], False, f"canonical_failed_status_{int(result['status_code'])}"


def token_length(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "get"):
        ids = ids.get("input_ids", [])
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def allocate_validation(items: list[dict[str, Any]], count: int) -> dict[str, int]:
    buckets = Counter(item["extra_info"]["interface_kind"] for item in items)
    exact = {key: count * value / len(items) for key, value in buckets.items()}
    allocated = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(allocated.values())
    order = sorted(exact, key=lambda key: exact[key] - allocated[key], reverse=True)
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    tests = [record["extra_info"]["test_count_used"] for record in records]
    prompts = [record["extra_info"]["prompt_tokens_max"] for record in records]
    return {
        "rows": len(records),
        "by_dataset": Counter(record["extra_info"]["dataset"] for record in records),
        "by_interface": Counter(record["extra_info"]["interface_kind"] for record in records),
        "by_difficulty": Counter(
            f"{record['extra_info']['dataset']}::{record['extra_info']['difficulty']}" for record in records
        ),
        "canonical_verified": sum(record["extra_info"]["canonical_verified"] for record in records),
        "structural_only": sum(not record["extra_info"]["canonical_verified"] for record in records),
        "tests_total": sum(tests),
        "tests_mean": statistics.mean(tests),
        "tests_median": statistics.median(tests),
        "tests_min": min(tests),
        "tests_max": max(tests),
        "prompt_tokens_mean": statistics.mean(prompts),
        "prompt_tokens_max": max(prompts),
    }


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reject_counts: Counter[str] = Counter()
    reject_examples: dict[str, list[str]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []

    for line in args.data_path.open(encoding="utf-8"):
        row = json.loads(line)
        meta, reason = normalize_row(row)
        if reason:
            reject_counts[reason] += 1
            if len(reject_examples[reason]) < 20:
                reject_examples[reason].append(str(row.get("id")))
            continue
        candidates.append({"row": row, "meta": meta})

    with ThreadPoolExecutor(max_workers=args.verify_workers) as executor:
        preflight = list(
            executor.map(
                lambda item: canonical_preflight(item, args.exec_timeout, args.per_test_timeout),
                candidates,
            )
        )
    status_by_id = {problem_id: (ok, reason) for problem_id, ok, reason in preflight}

    tokenizers = [
        AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        for path in args.model_paths
    ]
    accepted: list[dict[str, Any]] = []
    for item in candidates:
        problem_id = item["meta"]["problem_id"]
        ok, reason = status_by_id[problem_id]
        if not ok:
            reject_counts[reason] += 1
            if len(reject_examples[reason]) < 20:
                reject_examples[reason].append(problem_id)
            continue
        messages = render_messages(item["row"], item["meta"])
        lengths = [token_length(tokenizer, messages) for tokenizer in tokenizers]
        if max(lengths) > args.max_prompt_length:
            reject_counts["prompt_overlong"] += 1
            if len(reject_examples["prompt_overlong"]) < 20:
                reject_examples["prompt_overlong"].append(problem_id)
            continue
        canonical_verified = reason == "canonical_full_pass"
        accepted.append(
            {
                "prompt": messages,
                "data_source": f"stage3_{item['meta']['dataset']}_{item['meta']['interface_kind']}",
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(item["meta"], ensure_ascii=False),
                },
                "extra_info": {
                    "problem_id": problem_id,
                    "dataset": item["meta"]["dataset"],
                    "difficulty": item["meta"].get("difficulty") or "",
                    "mode": item["meta"]["mode"],
                    "interface_kind": item["meta"]["interface_kind"],
                    "test_count_used": len(item["meta"]["tests"]),
                    "canonical_verified": canonical_verified,
                    "prompt_tokens_by_model": dict(zip(map(str, args.model_paths), lengths, strict=True)),
                    "prompt_tokens_max": max(lengths),
                },
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(accepted)
    allocation = allocate_validation(accepted, args.val_count)
    val: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    used = Counter()
    for record in accepted:
        key = record["extra_info"]["interface_kind"]
        if used[key] < allocation[key]:
            val.append(record)
            used[key] += 1
        else:
            train.append(record)

    for index, record in enumerate(train):
        record["extra_info"]["index"] = index
    for index, record in enumerate(val):
        record["extra_info"]["index"] = index

    write_parquet(args.output_dir / "train.parquet", train)
    write_parquet(args.output_dir / "val.parquet", val)
    with (args.output_dir / "validation100.jsonl").open("w", encoding="utf-8") as handle:
        for record in val:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "seed": args.seed,
        "source": str(args.data_path),
        "model_paths": list(map(str, args.model_paths)),
        "all_tests_used": True,
        "max_prompt_length": args.max_prompt_length,
        "candidate_rows_after_structural_filter": len(candidates),
        "accepted_rows_before_holdout": len(accepted),
        "validation_allocation": allocation,
        "train": summarize(train),
        "validation": summarize(val),
        "reject_counts": reject_counts,
        "reject_examples": reject_examples,
        "reward": {
            "full_pass": 1.0,
            "partial": "0.8 * passed_tests / total_tests",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=dict),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
