#!/usr/bin/env python3
"""Summarize legacy checkpoint metrics without editing repository documents."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ORDER = [
    "Qwen2.5-7B-Instruct__base",
    "Qwen2.5-7B-Instruct__full_sft",
    "Qwen2.5-7B-Instruct__full_sft_14043",
    "Qwen2.5-7B-Instruct__no_math_number_theory",
    "Qwen2.5-7B-Instruct__no_data_structure",
    "Qwen2.5-7B-Instruct__no_dynamic_programming",
    "Qwen2.5-7B-Instruct__no_greedy_search",
    "Qwen2.5-7B-Instruct__no_implementation_simulation",
    "Qwen2.5-7B-Instruct__no_graph",
    "Qwen2.5-7B-Instruct__no_other_algorithm",
    "Qwen3-8B-Base__base",
    "Qwen3-8B-Base__full_sft",
    "Qwen3-8B-Base__full_sft_14043",
    "Qwen3-8B-Base__no_math_number_theory",
    "Qwen3-8B-Base__no_data_structure",
    "Qwen3-8B-Base__no_dynamic_programming",
    "Qwen3-8B-Base__no_greedy_search",
    "Qwen3-8B-Base__no_implementation_simulation",
    "Qwen3-8B-Base__no_graph",
    "Qwen3-8B-Base__no_other_algorithm",
    "stage3_dapo__qwen25_7b_final",
    "stage3_dapo__qwen3_8b_final",
    "stage3_dapo__qwen3_8b_step400_best_saved",
]

TASK_COLUMNS = [
    ("main", "arc_challenge", "ARC-C"),
    ("main", "finqa", "FinQA"),
    ("main", "humaneval", "HumanEval"),
    ("main", "legalbench", "LegalBench"),
    ("main", "math500_medium", "MATH med"),
    ("main", "mbpp_plus", "MBPP+"),
    ("main", "medcalc", "MedCalc"),
    ("hard", "apps_hard", "APPS hard"),
    ("hard", "math500_high_level", "MATH high"),
    ("diagnostic", "gsm8k", "GSM8K"),
    ("diagnostic", "mbpp_simple", "MBPP simple"),
    ("diagnostic", "scienceqa", "ScienceQA"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--model-order-file",
        type=Path,
        help="Optional JSON list replacing the archived 23-checkpoint display order.",
    )
    parser.add_argument(
        "--only-observed",
        action="store_true",
        help="Omit zero-run entries from the archived default model order.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Record unreadable metrics files instead of failing the summary.",
    )
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--title", default="Legacy Standard-Suite Checkpoint Evaluation")
    return parser.parse_args()


def metric_value(metric: Any) -> float | None:
    if not isinstance(metric, dict):
        return None
    value = metric.get("accuracy", metric.get("pass_at_1", metric.get("score")))
    return float(value) if value is not None else None


def stat(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def pct(summary: dict[str, float | int] | None) -> str:
    if summary is None:
        return "-"
    mean = 100 * float(summary["mean"])
    if int(summary["n"]) == 1:
        return f"{mean:.2f}%"
    return f"{mean:.2f}% +/- {100 * float(summary['population_std']):.2f}"


def seed_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return 0, f"{int(text):020d}"
    except ValueError:
        return 1, text


def load_runs(
    eval_root: Path, skip_invalid: bool
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    invalid: list[dict[str, str]] = []
    paths = sorted(eval_root.glob("seed_*/*/metrics.json"))
    if not paths:
        raise FileNotFoundError(
            f"no metrics matched {eval_root / 'seed_*' / '*' / 'metrics.json'}"
        )
    for path in paths:
        relative = path.relative_to(eval_root).as_posix()
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(metrics, dict):
                raise ValueError("top-level JSON value is not an object")
            model_name = str(metrics.get("model_name") or path.parent.name)
            metrics["_relative_path"] = relative
            metrics["_seed"] = metrics.get("seed", path.parents[1].name.removeprefix("seed_"))
            by_model.setdefault(model_name, []).append(metrics)
        except Exception as error:
            invalid.append({"path": relative, "error": f"{type(error).__name__}: {error}"})
    if invalid and not skip_invalid:
        details = "\n".join(f"- {item['path']}: {item['error']}" for item in invalid)
        raise ValueError(f"invalid metrics files:\n{details}")
    for rows in by_model.values():
        rows.sort(key=lambda item: seed_sort_key(item["_seed"]))
    return by_model, invalid


def task_values(
    rows: list[dict[str, Any]], suite: str, task: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        metric = row.get(suite, {}).get(task) if isinstance(row.get(suite), dict) else None
        value = metric_value(metric)
        if value is not None:
            values.append(value)
    return values


def top_level_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def nested_values(rows: list[dict[str, Any]], suite: str, key: str) -> list[float]:
    values = []
    for row in rows:
        section = row.get(suite)
        if isinstance(section, dict) and section.get(key) is not None:
            values.append(float(section[key]))
    return values


def display_name(model_name: str) -> str:
    if "__" not in model_name:
        return model_name
    family, arm = model_name.split("__", 1)
    return f"{family} / {arm}"


def load_model_order(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_MODEL_ORDER)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("--model-order-file must contain a JSON list of model-name strings")
    if len(set(payload)) != len(payload):
        raise ValueError("--model-order-file contains duplicate model names")
    return payload


def resolved_order(
    configured: list[str], observed: set[str], only_observed: bool
) -> list[str]:
    ordered = [model for model in configured if not only_observed or model in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def aggregate_model(model_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_stats = {
        task: stat(task_values(rows, suite, task))
        for suite, task, _ in TASK_COLUMNS
    }
    return {
        "model_name": model_name,
        "display_name": display_name(model_name),
        "run_count": len(rows),
        "seeds": [row["_seed"] for row in rows],
        "source_metrics": [row["_relative_path"] for row in rows],
        "scoreable_overall": stat(top_level_values(rows, "scoreable_overall_mean")),
        "main_scoreable": stat(nested_values(rows, "main", "main_scoreable_mean")),
        "hard_mean": stat(nested_values(rows, "hard", "hard_mean")),
        "diagnostic_mean": stat(
            nested_values(rows, "diagnostic", "diagnostic_mean")
        ),
        "tasks": task_stats,
    }


def render_markdown(
    title: str,
    eval_root: Path,
    aggregates: list[dict[str, Any]],
    run_count: int,
    invalid: list[dict[str, str]],
) -> str:
    observed = [row for row in aggregates if row["run_count"]]
    seeds = sorted(
        {str(seed) for row in observed for seed in row["seeds"]}, key=seed_sort_key
    )
    lines = [
        f"# {title}\n\n",
        f"- Metrics tree: `{eval_root.name}`\n",
        f"- Parsed runs: {run_count}\n",
        f"- Observed models/checkpoints: {len(observed)}\n",
        f"- Seeds: {', '.join(seeds) if seeds else '-'}\n",
        "- Standard deviation: population standard deviation across available runs.\n",
        "- HealthBench is not part of this legacy scoreable aggregate.\n",
    ]
    if invalid:
        lines.append(f"- Invalid files skipped: {len(invalid)} (see JSON output for details).\n")

    lines.extend(
        [
            "\n## Aggregate summary\n\n",
            "| Model / arm | Runs | Scoreable overall | Main scoreable | Hard mean | Diagnostic mean | HumanEval | MBPP+ | APPS hard |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in aggregates:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["display_name"],
                    str(row["run_count"]),
                    pct(row["scoreable_overall"]),
                    pct(row["main_scoreable"]),
                    pct(row["hard_mean"]),
                    pct(row["diagnostic_mean"]),
                    pct(row["tasks"]["humaneval"]),
                    pct(row["tasks"]["mbpp_plus"]),
                    pct(row["tasks"]["apps_hard"]),
                ]
            )
            + " |\n"
        )

    labels = [label for _, _, label in TASK_COLUMNS]
    lines.extend(
        [
            "\n## Per-task aggregate\n\n",
            "| Model / arm | Runs | " + " | ".join(labels) + " |\n",
            "|---|---:|" + "|".join(["---:"] * len(labels)) + "|\n",
        ]
    )
    for row in aggregates:
        values = [pct(row["tasks"][task]) for _, task, _ in TASK_COLUMNS]
        lines.append(
            "| "
            + " | ".join([row["display_name"], str(row["run_count"]), *values])
            + " |\n"
        )
    return "".join(lines)


def write_text(path: Path, content: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    if not eval_root.is_dir():
        raise FileNotFoundError(f"--eval-root is not a directory: {eval_root}")
    by_model, invalid = load_runs(eval_root, args.skip_invalid)
    run_count = sum(len(rows) for rows in by_model.values())
    if args.expected_runs is not None and run_count != args.expected_runs:
        raise RuntimeError(f"expected {args.expected_runs} parsed runs, found {run_count}")

    configured = load_model_order(args.model_order_file)
    order = resolved_order(configured, set(by_model), args.only_observed)
    aggregates = [aggregate_model(model, by_model.get(model, [])) for model in order]
    report = {
        "schema_version": 1,
        "contract": "legacy_standard_checkpoint_summary",
        "metrics_root_name": eval_root.name,
        "run_count": run_count,
        "observed_model_count": len(by_model),
        "invalid_files": invalid,
        "model_order": order,
        "models": aggregates,
        "task_columns": [
            {"suite": suite, "task": task, "label": label}
            for suite, task, label in TASK_COLUMNS
        ],
    }
    write_text(
        args.output_md,
        render_markdown(args.title, eval_root, aggregates, run_count, invalid),
    )
    if args.output_json is not None:
        write_text(
            args.output_json,
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
    print(
        json.dumps(
            {
                "output_md": str(args.output_md),
                "output_json": str(args.output_json) if args.output_json else None,
                "run_count": run_count,
                "observed_model_count": len(by_model),
                "invalid_files": len(invalid),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
