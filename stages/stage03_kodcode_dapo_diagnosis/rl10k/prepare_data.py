#!/usr/bin/env python3
"""Prepare verified KodCode RL-10K data for the four-model DAPO diagnosis."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


STAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STAGE_ROOT.parents[1]
sys.path.insert(0, str(STAGE_ROOT / "shared"))

from frozen_contract import generated_contract, load_contract, require_equal, sha256_file


def env_path(name: str, *parts: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().joinpath(*parts) if value else None
SAFE_EXTERNAL_MODULES = {
    "pytest",
    "numpy",
    "pandas",
    "torch",
    "sklearn",
    "scipy",
    "sympy",
    "PIL",
    "requests",
}
DISALLOWED_SIDE_EFFECT_MODULES = {"subprocess", "multiprocessing", "socket", "requests"}
TEST_CODE_REPAIRS = {
    "rl10k_r1::Filter_60220_I": {
        "reason": "replace wall-clock cache timing assertion with deterministic cache_info hit check",
        "test_code": """from solution import fibonacci

def test_fibonacci_base_cases():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1

def test_fibonacci_small_numbers():
    assert fibonacci(2) == 1
    assert fibonacci(3) == 2
    assert fibonacci(4) == 3
    assert fibonacci(5) == 5
    assert fibonacci(6) == 8

def test_fibonacci_large_numbers():
    assert fibonacci(50) == 12586269025
    assert fibonacci(100) == 354224848179261915075

def test_fibonacci_cached():
    assert hasattr(fibonacci, "cache_info")
    assert hasattr(fibonacci, "cache_clear")
    fibonacci.cache_clear()
    fibonacci(100)
    hits_before = fibonacci.cache_info().hits
    fibonacci(100)
    assert fibonacci.cache_info().hits == hits_before + 1
""",
    },
    "rl10k_r1::Filter_1152_I": {
        "reason": "seed Python random before category assertions to remove stochastic pass/fail",
        "test_code": """from solution import generate_password
import random
import string

def _assert_password(password, length):
    assert len(password) == length
    assert any(c.islower() for c in password)
    assert any(c.isupper() for c in password)
    assert any(c.isdigit() for c in password)
    assert any(c in string.punctuation for c in password)
    assert all(c in (string.ascii_letters + string.digits + string.punctuation) for c in password)

def test_length_less_than_8():
    assert generate_password(7) is None

def test_minimum_length():
    random.seed(0)
    _assert_password(generate_password(8), 8)

def test_length_greater_than_minimum():
    random.seed(0)
    _assert_password(generate_password(12), 12)

def test_verify_characters():
    random.seed(0)
    password = generate_password(10)
    assert len(password) == 10
    assert all(c in (string.ascii_letters + string.digits + string.punctuation) for c in password)
""",
    },
    "rl10k_r1::Filter_80484_I": {
        "reason": "compare unordered intersection semantics instead of hash-dependent list order",
        "test_code": """from solution import intersection

def _assert_intersection(actual, expected):
    assert set(actual) == set(expected)
    assert len(actual) == len(set(actual))

def test_intersection_with_common_elements():
    _assert_intersection(intersection([1, 2, 3, 4], [3, 4, 5, 6]), [3, 4])

def test_intersection_with_no_common_elements():
    _assert_intersection(intersection([1, 2, 3], [4, 5, 6]), [])

def test_intersection_with_all_common_elements():
    _assert_intersection(intersection([1, 2, 3], [1, 2, 3]), [1, 2, 3])

def test_intersection_with_empty_lists():
    _assert_intersection(intersection([], []), [])

def test_intersection_with_one_empty_list():
    _assert_intersection(intersection([1, 2, 3], []), [])
    _assert_intersection(intersection([], [1, 2, 3]), [])

def test_intersection_with_duplicate_elements():
    _assert_intersection(intersection([1, 1, 2, 2], [2, 2, 3, 3]), [2])

def test_intersection_with_mixed_types():
    _assert_intersection(intersection([1, "2", 3.0], [3.0, "2", 4]), [3.0, "2"])
""",
    },
}


def parse_args() -> argparse.Namespace:
    default_source = env_path("CG_DATA_ROOT", "kodcode4o_r1_clean38k", "train.parquet")
    default_output = env_path(
        "CG_OUTPUT_ROOT", "stage03_kodcode_dapo_diagnosis", "rl10k", "data"
    )
    default_model_root = env_path("CG_MODEL_ROOT")
    default_data_test_root = env_path("CG_DATA_TEST_ROOT")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=default_source, required=default_source is None
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_output, required=default_output is None
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=default_model_root,
        required=default_model_root is None,
    )
    parser.add_argument(
        "--data-test-root",
        type=Path,
        default=default_data_test_root,
        required=default_data_test_root is None,
    )
    parser.add_argument(
        "--plain-template",
        type=Path,
        default=STAGE_ROOT / "plain_chat_template.jinja",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--validation-rows", type=int, default=100)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/stage03_kodcode_dapo_diagnosis.json",
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="Warn instead of failing when a frozen source count/hash differs.",
    )
    parser.add_argument(
        "--reward-preflight-rejects",
        type=Path,
        help="Optional JSON containing exact-reward failure_ids from a prior pass.",
    )
    args = parser.parse_args()
    if args.reward_preflight_rejects is None:
        args.reward_preflight_rejects = (
            args.output_dir / "audit/exact_reward_initial_failures.json"
        )
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", str(text).lower())


def ngrams(words: list[str], n: int = 5) -> set[tuple[str, ...]]:
    return {tuple(words[index : index + n]) for index in range(max(0, len(words) - n + 1))}


def formal_text(row: dict[str, Any]) -> str:
    for key in ("text", "prompt", "problem", "question", "description"):
        if row.get(key):
            return str(row[key])
    return ""


def load_formal_ngrams(formal_files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in formal_files:
        for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            text = formal_text(row)
            grams = ngrams(normalize_words(text))
            if not grams:
                continue
            records.append(
                {
                    "dataset": path.stem,
                    "id": row.get("task_id") or row.get("id") or line_index,
                    "grams": grams,
                    "normalized": " ".join(normalize_words(text)),
                }
            )
    return records


def overlap_match(prompt: str, formal_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    words = normalize_words(prompt)
    normalized = " ".join(words)
    prompt_grams = ngrams(words)
    if not prompt_grams:
        return None
    best: dict[str, Any] | None = None
    for formal in formal_records:
        exact = normalized == formal["normalized"]
        overlap = len(prompt_grams & formal["grams"])
        coverage = overlap / max(1, len(formal["grams"]))
        jaccard = overlap / max(1, len(prompt_grams | formal["grams"]))
        if exact or (overlap >= 8 and coverage >= 0.85):
            candidate = {
                "dataset": formal["dataset"],
                "formal_id": formal["id"],
                "exact": exact,
                "coverage": coverage,
                "jaccard": jaccard,
            }
            if best is None or (coverage, jaccard) > (best["coverage"], best["jaccard"]):
                best = candidate
    return best


def inspect_tests(test_code: str) -> tuple[list[str] | None, str | None, list[str]]:
    try:
        tree = ast.parse(test_code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None, "test_code_parse_failure", []
    tests = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test")
    ]
    if not tests:
        return None, "no_top_level_test_functions", []
    if any(
        node.args.args
        or node.args.posonlyargs
        or node.args.kwonlyargs
        or node.args.vararg
        or node.args.kwarg
        for node in tests
    ):
        return None, "pytest_fixture_or_parameter_required", []
    if any(node.decorator_list for node in tests):
        return None, "decorated_or_parameterized_test", []
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".", 1)[0])
    modules.discard("solution")
    unsupported = sorted(
        module
        for module in modules
        if module not in sys.stdlib_module_names and module not in SAFE_EXTERNAL_MODULES
    )
    if unsupported:
        return None, f"unsupported_test_dependency:{','.join(unsupported)}", sorted(modules)
    risky = sorted(modules & DISALLOWED_SIDE_EFFECT_MODULES)
    if risky:
        return None, f"nondeterministic_or_external_side_effect:{','.join(risky)}", sorted(modules)
    return [node.name for node in tests], None, sorted(modules)


def entry_point(row: dict[str, Any]) -> str:
    value = row.get("test_info")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return str(value[0].get("function_name") or "")
    return ""


def render_user_prompt(row: dict[str, Any], test_count: int) -> str:
    return (
        "Solve the following Python programming task.\n\n"
        "STRICT OUTPUT CONTRACT:\n"
        "- Return only one complete executable Python solution.\n"
        "- Define the required function exactly as declared in the problem.\n"
        "- Do not include Markdown fences, explanations, local tests, assertions, or diagnostics.\n"
        "- The hidden verifier contains "
        f"{test_count} independent test functions; your solution must pass all of them.\n\n"
        f"{str(row['prompt']).strip()}\n"
    )


def proportional_validation(records: list[dict[str, Any]], count: int, seed: int) -> tuple[list, list]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        extra = record["extra_info"]
        buckets[(extra["subset"], extra["difficulty"])].append(record)
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    exact = {key: count * len(rows) / len(records) for key, rows in buckets.items()}
    allocated = {key: min(len(buckets[key]), int(math.floor(value))) for key, value in exact.items()}
    remaining = count - sum(allocated.values())
    order = sorted(
        buckets,
        key=lambda key: (exact[key] - allocated[key], len(buckets[key]), key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if allocated[key] < len(buckets[key]):
                allocated[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate validation rows")
    validation: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for key, rows in buckets.items():
        split = allocated[key]
        validation.extend(rows[:split])
        train.extend(rows[split:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def token_lengths(
    records: list[dict[str, Any]],
    model_specs: dict[str, tuple[Path, str]],
    plain_template: str,
    max_prompt_length: int,
) -> tuple[dict[str, dict[str, int]], set[str]]:
    lengths: dict[str, dict[str, int]] = {}
    overlong: set[str] = set()
    for model_key, (model_path, mode) in model_specs.items():
        print(f"[token-audit] loading {model_key}: {model_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        model_lengths: dict[str, int] = {}
        for index, record in enumerate(records):
            kwargs: dict[str, Any] = {}
            if mode == "plain":
                kwargs["chat_template"] = plain_template
            elif model_key.startswith("qwen"):
                kwargs["enable_thinking"] = False
            ids = tokenizer.apply_chat_template(
                record["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                **kwargs,
            )
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            length = len(ids)
            model_lengths[record["extra_info"]["problem_id"]] = length
            if length > max_prompt_length:
                overlong.add(record["extra_info"]["problem_id"])
            if (index + 1) % 2500 == 0:
                print(f"[token-audit] {model_key} {index + 1}/{len(records)}", flush=True)
        lengths[model_key] = model_lengths
    return lengths, overlong


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    tests = [int(row["extra_info"]["test_count_used"]) for row in records]
    return {
        "rows": len(records),
        "test_functions_total": sum(tests),
        "test_functions_mean": statistics.mean(tests) if tests else 0,
        "test_functions_median": statistics.median(tests) if tests else 0,
        "test_functions_min": min(tests) if tests else 0,
        "test_functions_max": max(tests) if tests else 0,
        "subset": dict(Counter(row["extra_info"]["subset"] for row in records)),
        "difficulty": dict(Counter(row["extra_info"]["difficulty"] for row in records)),
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def main() -> None:
    args = parse_args()
    formal_files = [
        args.data_test_root / "main_test/humaneval.jsonl",
        args.data_test_root / "main_test/mbpp_plus.jsonl",
        args.data_test_root / "diagnostic_test/mbpp_simple.jsonl",
    ]
    model_specs = {
        "qwen25_native": (args.model_root / "Qwen2.5-7B-Instruct", "native"),
        "qwen3_plain": (args.model_root / "Qwen3-8B-Base", "plain"),
        "llama31_native": (args.model_root / "Llama-3.1-8B-Instruct", "native"),
        "gemma2_native": (args.model_root / "Gemma-2-9B-Instruct", "native"),
    }
    plain_template = args.plain_template.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = args.output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(args.source)
    source = source[source["source_dataset"].eq("KodCode-Light-RL-10K")].copy()
    frozen, branch_contract = load_contract(args.config, "rl10k")
    require_equal(
        "source_parquet_sha256",
        sha256_file(args.source),
        frozen["source_parquet_sha256"],
        exploratory=args.exploratory,
    )
    require_equal(
        "rl10k.input_rows",
        len(source),
        branch_contract["input_rows"],
        exploratory=args.exploratory,
    )
    formal_records = load_formal_ngrams(formal_files)
    reward_preflight_rejects: set[str] = set()
    if args.reward_preflight_rejects.is_file():
        prior = json.loads(args.reward_preflight_rejects.read_text(encoding="utf-8"))
        reward_preflight_rejects = {str(value) for value in prior.get("failure_ids", [])}
    rejections: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    applied_test_repairs: list[dict[str, str]] = []

    for row in source.to_dict(orient="records"):
        problem_id = str(row["id"])
        if not str(row.get("prompt") or "").strip() or not str(row.get("test_code") or "").strip():
            rejections.append({"problem_id": problem_id, "reason": "missing_prompt_or_tests"})
            continue
        if row.get("quality_status") != "official_rl10k_r1_correct_and_test_verified;local_full_execution_passed":
            rejections.append({"problem_id": problem_id, "reason": "not_full_execution_verified"})
            continue
        if problem_id in reward_preflight_rejects:
            rejections.append(
                {
                    "problem_id": problem_id,
                    "reason": "exact_reward_reference_not_full_pass",
                }
            )
            continue
        test_code = str(row["test_code"])
        test_repair = TEST_CODE_REPAIRS.get(problem_id)
        if test_repair:
            test_code = test_repair["test_code"]
            applied_test_repairs.append(
                {"problem_id": problem_id, "reason": test_repair["reason"]}
            )
        test_names, reason, modules = inspect_tests(test_code)
        if reason:
            rejections.append({"problem_id": problem_id, "reason": reason})
            continue
        assert test_names is not None
        overlap = overlap_match(str(row["prompt"]), formal_records)
        if overlap:
            rejections.append(
                {
                    "problem_id": problem_id,
                    "reason": "formal_code_benchmark_near_duplicate",
                    **overlap,
                }
            )
            continue
        function_name = entry_point(row)
        if not function_name:
            rejections.append({"problem_id": problem_id, "reason": "missing_entry_point"})
            continue
        messages = [{"role": "user", "content": render_user_prompt(row, len(test_names))}]
        ground_truth = {
            "problem_id": problem_id,
            "dataset": "KodCode-Light-RL-10K",
            "interface_kind": "kodcode_pytest_function",
            "entry_point": function_name,
            "test_code": test_code,
            "test_functions": test_names,
            "test_count": len(test_names),
        }
        records.append(
            {
                "prompt": messages,
                "data_source": "diagnose_kodcode_rl10k",
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
                },
                "extra_info": {
                    "problem_id": problem_id,
                    "original_question_id": str(row["original_question_id"]),
                    "subset": str(row["subset"]),
                    "difficulty": str(row["difficulty"]),
                    "style": str(row["style"]),
                    "entry_point": function_name,
                    "test_count_used": len(test_names),
                    "test_modules": modules,
                    "test_code_repaired": bool(test_repair),
                    "source_full_execution_verified": True,
                },
            }
        )

    lengths, overlong_ids = token_lengths(
        records, model_specs, plain_template, args.max_prompt_length
    )
    if overlong_ids:
        kept = []
        for record in records:
            problem_id = record["extra_info"]["problem_id"]
            if problem_id in overlong_ids:
                rejections.append(
                    {
                        "problem_id": problem_id,
                        "reason": f"prompt_over_{args.max_prompt_length}_tokens",
                    }
                )
            else:
                kept.append(record)
        records = kept

    # Recompute per-row token metadata after any overlength rejection.
    original_ids = [record["extra_info"]["problem_id"] for record in records]
    for record in records:
        problem_id = record["extra_info"]["problem_id"]
        record["extra_info"]["prompt_tokens_by_model"] = {
            key: values[problem_id] for key, values in lengths.items()
        }
        record["extra_info"]["prompt_tokens_max"] = max(
            record["extra_info"]["prompt_tokens_by_model"].values()
        )

    # All retained rows have unique source IDs; keep the check explicit.
    if len(original_ids) != len(set(original_ids)):
        raise RuntimeError("duplicate RL problem IDs remain")
    train, validation = proportional_validation(records, args.validation_rows, args.seed)
    for index, record in enumerate(train):
        record["extra_info"]["index"] = index
    for index, record in enumerate(validation):
        record["extra_info"]["index"] = index
    train_ids = {row["extra_info"]["problem_id"] for row in train}
    val_ids = {row["extra_info"]["problem_id"] for row in validation}
    if train_ids & val_ids:
        raise RuntimeError("train/validation overlap")

    write_parquet(args.output_dir / "train.parquet", train)
    write_parquet(args.output_dir / "val.parquet", validation)
    with (args.output_dir / "validation100.jsonl").open("w", encoding="utf-8") as handle:
        for record in validation:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (audit_dir / "rejections.jsonl").open("w", encoding="utf-8") as handle:
        for record in rejections:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (audit_dir / "test_repairs.json").write_text(
        json.dumps(
            {
                "actual_repaired": len(applied_test_repairs),
                "repairs": applied_test_repairs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    token_summary = {}
    for key in lengths:
        retained_values = [
            int(row["extra_info"]["prompt_tokens_by_model"][key]) for row in records
        ]
        token_summary[key] = {
            "mean": statistics.mean(retained_values),
            "median": statistics.median(retained_values),
            "p95": sorted(retained_values)[max(0, math.ceil(0.95 * len(retained_values)) - 1)],
            "max": max(retained_values),
        }
    manifest = {
        "version": "diagnose_dapo_kodcode_rl10k_v1",
        "seed": args.seed,
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "input_verified_rl_rows": len(source),
        "accepted_before_split": len(records),
        "rejected": len(rejections),
        "rejection_reasons": dict(Counter(row["reason"] for row in rejections)),
        "train": summarize(train),
        "validation": summarize(validation),
        "train_validation_id_overlap": 0,
        "formal_overlap_reference_rows": len(formal_records),
        "formal_near_duplicate_filter": {
            "method": "normalized exact match or high five-gram lexical coverage",
            "reference_rows": len(formal_records),
            "actual_rejected": sum(
                row["reason"] == "formal_code_benchmark_near_duplicate"
                for row in rejections
            ),
        },
        "deterministic_test_repairs": {
            "actual_repaired": len(applied_test_repairs),
            "audit_file": str(audit_dir / "test_repairs.json"),
        },
        "exact_reward_preflight_reject_file": (
            str(args.reward_preflight_rejects)
            if args.reward_preflight_rejects.is_file()
            else None
        ),
        "prompt_token_summary": token_summary,
        "prompt_contracts": {
            "qwen25": "native tokenizer chat template; enable_thinking=false",
            "qwen3": "plain raw user content via local-copy tokenizer template",
            "llama31": "native tokenizer chat template",
            "gemma2": "native tokenizer chat template; one user turn and no system role",
        },
        "reward_contract": {
            "test_unit": "one top-level zero-argument test_* function",
            "all_pass": 1.0,
            "any_failure": 0.0,
            "pass_fraction_is_diagnostic_only": True,
            "tests_hidden_from_prompt": True,
        },
        "files": {
            "train": str(args.output_dir / "train.parquet"),
            "validation": str(args.output_dir / "val.parquet"),
            "validation_jsonl": str(args.output_dir / "validation100.jsonl"),
            "rejections": str(audit_dir / "rejections.jsonl"),
        },
        "generated_contract": generated_contract(
            args.output_dir / "train.parquet",
            args.output_dir / "val.parquet",
            train,
            validation,
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
