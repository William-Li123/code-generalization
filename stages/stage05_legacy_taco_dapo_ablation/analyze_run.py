#!/usr/bin/env python3
"""Summarize a Stage 3 verl/DAPO smoke run."""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from stage3_code_reward import extract_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-response-length", type=int, required=True)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * q))))
    return float(xs[idx])


def maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def optional_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    xs = sorted(float(value) for value in values)
    n = len(xs)
    weighted = sum((index + 1) * value for index, value in enumerate(xs))
    return (2.0 * weighted) / (n * sum(xs)) - (n + 1.0) / n


def count_distribution(values: list[int]) -> dict[str, Any]:
    histogram = collections.Counter(values)
    return {
        "count": len(values),
        "sum": sum(values),
        "min": min(values, default=0),
        "max": max(values, default=0),
        "mean": statistics.mean(values) if values else 0.0,
        "gini": gini(values),
        "more_than_once": sum(value > 1 for value in values),
        "histogram": {str(key): histogram[key] for key in sorted(histogram)},
    }


def row_passed(row: dict[str, Any]) -> float:
    if "passed" in row:
        return float(row.get("passed") or 0.0)
    return float(row.get("score") or 0.0)


def row_format_ok(row: dict[str, Any]) -> float:
    if "format_ok" in row:
        return float(row.get("format_ok") or 0.0)
    text = (row.get("output") or "").strip()
    if not text or "```" in text:
        return 0.0
    prose_markers = ["Here is", "Explanation", "The solution"]
    return float(not any(marker.lower() in text[:160].lower() for marker in prose_markers))


def row_parse_ok(row: dict[str, Any]) -> float:
    if "parse_ok" in row:
        return float(row.get("parse_ok") or 0.0)
    ground_truth = maybe_json(row.get("gts"))
    starter_code = ground_truth.get("starter_code", "") if isinstance(ground_truth, dict) else ""
    try:
        ast.parse(extract_code(row.get("output") or "", starter_code))
        return 1.0
    except SyntaxError:
        return 0.0


def read_rollouts(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "rollouts").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    item["_file"] = path.name
                    rows.append(item)
    return rows


def read_peak_memory(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "memory.csv"
    if not path.exists():
        return {"peak_memory_mib": 0, "num_memory_samples": 0}
    peak = 0.0
    per_gpu_peak: dict[str, float] = {}
    samples = 0
    with path.open(encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            samples += 1
            used_columns = [
                key
                for key in row
                if key and (key.endswith("_used_mib") or key in {"memory.used [MiB]", "memory.used"})
            ]
            for key in used_columns:
                try:
                    value = float(row.get(key) or 0)
                except Exception:
                    continue
                peak = max(peak, value)
                per_gpu_peak[key] = max(per_gpu_peak.get(key, 0.0), value)
    return {
        "peak_memory_mib": int(peak),
        "peak_memory_gib": peak / 1024 if peak else 0,
        "per_gpu_peak_memory_mib": {key: int(value) for key, value in sorted(per_gpu_peak.items())},
        "num_memory_samples": samples,
    }


def parse_log_metrics(run_dir: Path) -> dict[str, float]:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    keys = [
        "response_length/mean",
        "response_length/max",
        "response_length/clip_ratio",
        "actor/pg_loss",
        "actor/pg_clipfrac",
        "critic/score/mean",
        "critic/rewards/mean",
    ]
    out: dict[str, float] = {}
    for key in keys:
        pattern = re.compile(re.escape(key) + r"['\"]?\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            try:
                out[key.replace("/", "_")] = float(matches[-1])
            except Exception:
                pass
    return out


def read_utilization(run_dir: Path) -> dict[str, Any]:
    events_path = run_dir / "utilization_events.jsonl"
    summary_path = run_dir / "utilization_summary.json"
    events = []
    if events_path.exists():
        with events_path.open(encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
    generation = [event for event in events if event.get("event") == "generation_batch"]
    optimizer = [event for event in events if event.get("event") == "optimizer_step"]
    generated_groups = sum(int(event.get("generated_groups", 0)) for event in generation)
    valid_groups = sum(int(event.get("valid_groups", 0)) for event in generation)
    completion_count = sum(int(event.get("completion_count", 0)) for event in generation)
    response_tokens = sum(int(event.get("response_tokens_sum", 0)) for event in generation)
    truncated = sum(int(event.get("truncated_count", 0)) for event in generation)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    generated_counts = list((summary.get("generated") or {}).values())
    valid_counts = list((summary.get("valid") or {}).values())
    trained_counts = list((summary.get("trained") or {}).values())
    buffered_counts = list((summary.get("buffered") or {}).values())
    group_reward_means = [
        float(value)
        for event in generation
        for value in event.get("group_reward_means", [])
    ]
    group_reward_stds = [
        float(value)
        for event in generation
        for value in event.get("group_reward_stds", [])
    ]
    return {
        "generation_batches": len(generation),
        "optimizer_steps": len(optimizer),
        "generated_groups": generated_groups,
        "valid_groups": valid_groups,
        "raw_valid_group_rate": valid_groups / max(generated_groups, 1),
        "raw_completion_count": completion_count,
        "raw_response_tokens_mean": response_tokens / max(completion_count, 1),
        "raw_truncated_count": truncated,
        "raw_truncation_rate": truncated / max(completion_count, 1),
        "generated_unique_prompts": summary.get("generated_unique", 0),
        "valid_unique_prompts": summary.get("valid_unique", 0),
        "trained_unique_prompts": summary.get("trained_unique", 0),
        "remaining_buffer_groups": summary.get("remaining_buffer_groups", 0),
        "generated_count_distribution": count_distribution(generated_counts),
        "valid_count_distribution": count_distribution(valid_counts),
        "trained_count_distribution": count_distribution(trained_counts),
        "buffered_count_distribution": count_distribution(buffered_counts),
        "group_reward_mean": optional_mean(group_reward_means),
        "group_reward_mean_p10": percentile(group_reward_means, 0.10),
        "group_reward_mean_p50": percentile(group_reward_means, 0.50),
        "group_reward_mean_p90": percentile(group_reward_means, 0.90),
        "group_reward_std_mean": optional_mean(group_reward_stds),
        "all_equal_group_rate": (
            sum(value <= 1e-8 for value in group_reward_stds) / len(group_reward_stds)
            if group_reward_stds
            else 0.0
        ),
        "valid_rate_by_generation_batch": [event.get("valid_rate") for event in generation],
        "carry_groups_by_step": [event.get("carry_groups") for event in optimizer],
    }


def read_validation_curve(run_dir: Path) -> list[dict[str, Any]]:
    curve = []
    for path in sorted((run_dir / "validation").glob("*.jsonl"), key=lambda item: int(item.stem)):
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if not rows:
            continue
        curve.append(
            {
                "step": int(path.stem),
                "n": len(rows),
                "reward_mean": statistics.mean(float(row.get("score") or 0.0) for row in rows),
                "full_pass_rate": statistics.mean(
                    float(row.get("passed", float(row.get("score") == 1.0))) for row in rows
                ),
                "test_pass_fraction_mean": optional_mean(
                    [float(row["pass_frac"]) for row in rows if "pass_frac" in row]
                ),
                "parse_ok_rate": statistics.mean([row_parse_ok(row) for row in rows]),
                "format_ok_rate": statistics.mean([row_format_ok(row) for row in rows]),
            }
        )
    return curve


def summarize(rows: list[dict[str, Any]], tokenizer: Any, max_response_length: int) -> dict[str, Any]:
    if not rows:
        return {"num_completions": 0}

    rewards = [float(row.get("score") or 0.0) for row in rows]
    lengths = [len(tokenizer(row.get("output") or "", add_special_tokens=False)["input_ids"]) for row in rows]

    by_problem: dict[str, list[dict[str, Any]]] = {}
    by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        gt = maybe_json(row.get("gts"))
        problem_id = gt.get("problem_id") if isinstance(gt, dict) else str(row.get("input", ""))[:80]
        dataset = gt.get("dataset") if isinstance(gt, dict) else "unknown"
        by_problem.setdefault(str(problem_id), []).append(row)
        bucket = by_dataset.setdefault(str(dataset), {"n": 0, "mean_score": 0.0, "pass_rate": 0.0})
        bucket["n"] += 1
        bucket["mean_score"] += float(row.get("score") or 0.0)
        bucket["pass_rate"] += row_passed(row)

    group_stds: list[float] = []
    nonzero_groups = 0
    all_zero_groups = 0
    all_one_groups = 0
    all_equal_groups = 0
    any_pass_groups = 0
    for group in by_problem.values():
        rs = [float(row.get("score") or 0.0) for row in group]
        if len(rs) >= 2:
            std = statistics.pstdev(rs)
            group_stds.append(std)
            nonzero_groups += int(std > 1e-8)
            all_equal_groups += int(std <= 1e-8)
        all_zero_groups += int(all(r <= 1e-8 for r in rs))
        all_one_groups += int(all(r >= 0.999 for r in rs))
        any_pass_groups += int(any(row_passed(row) > 0.5 for row in group))

    for bucket in by_dataset.values():
        n = max(int(bucket["n"]), 1)
        bucket["mean_score"] /= n
        bucket["pass_rate"] /= n

    return {
        "num_completions": len(rows),
        "num_prompts": len(by_problem),
        "mean_reward": statistics.mean(rewards),
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_group_reward_std": statistics.mean(group_stds) if group_stds else 0.0,
        "valid_group_rate": nonzero_groups / max(len(by_problem), 1),
        "all_equal_group_rate": all_equal_groups / max(len(by_problem), 1),
        "all_zero_group_rate": all_zero_groups / max(len(by_problem), 1),
        "all_one_group_rate": all_one_groups / max(len(by_problem), 1),
        "any_pass_group_rate": any_pass_groups / max(len(by_problem), 1),
        "format_ok_rate": statistics.mean([row_format_ok(row) for row in rows]),
        "parse_ok_rate": statistics.mean([row_parse_ok(row) for row in rows]),
        "interface_ok_rate": optional_mean(
            [float(row["interface_ok"]) for row in rows if "interface_ok" in row]
        ),
        "pass_rate": statistics.mean([row_passed(row) for row in rows]),
        "pass_frac_mean": optional_mean([float(row["pass_frac"]) for row in rows if "pass_frac" in row]),
        "reward_diagnostics_present_rate": sum("passed" in row for row in rows) / len(rows),
        "completion_tokens_mean": statistics.mean(lengths),
        "completion_tokens_p50": percentile(lengths, 0.50),
        "completion_tokens_p90": percentile(lengths, 0.90),
        "completion_tokens_p95": percentile(lengths, 0.95),
        "completion_tokens_p99": percentile(lengths, 0.99),
        "completion_tokens_max": max(lengths),
        "clipped_ratio_proxy": sum(int(x >= max_response_length - 1) for x in lengths) / max(len(lengths), 1),
        "by_dataset": by_dataset,
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    rows = read_rollouts(args.run_dir)
    summary = {
        "run_dir": str(args.run_dir),
        "max_response_length": args.max_response_length,
        **read_peak_memory(args.run_dir),
        "rollout": summarize(rows, tokenizer, args.max_response_length),
        "utilization": read_utilization(args.run_dir),
        "validation_curve": read_validation_curve(args.run_dir),
        "log_metrics": parse_log_metrics(args.run_dir),
    }
    out = args.run_dir / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
