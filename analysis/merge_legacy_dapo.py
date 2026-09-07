#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MAIN_TASKS = [
    "arc_challenge",
    "finqa",
    "humaneval",
    "legalbench",
    "math500_medium",
    "mbpp_plus",
    "medcalc",
]
HARD_TASKS = ["math500_high_level"]
DIAGNOSTIC_TASKS = ["gsm8k", "mbpp_simple", "scienceqa"]
ALL_TASKS = MAIN_TASKS + HARD_TASKS + DIAGNOSTIC_TASKS
EXCLUDED = {
    "apps_hard": "excluded by user request",
    "planbench_hard_all_tasks": "excluded by user request",
    "healthbench": "excluded by user request",
    "healthbench_professional": "excluded by user request",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-spec", action="append", required=True, help="label|base|adapter")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metric: dict[str, Any] | None) -> float | None:
    if not isinstance(metric, dict):
        return None
    for key in ["accuracy", "pass_at_1", "score"]:
        value = metric.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def parse_model_specs(specs: list[str]) -> list[dict[str, str]]:
    models = []
    for spec in specs:
        parts = spec.split("|", 2)
        if len(parts) != 3:
            raise SystemExit(f"Bad --model-spec {spec!r}; expected label|base|adapter")
        models.append({"label": parts[0], "base_model_path": parts[1], "adapter_path": parts[2]})
    return models


def merge_group_metrics(run_dir: Path, label: str, model: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model_name": label,
        "base_model_path": model["base_model_path"],
        "adapter_path": model["adapter_path"],
        "excluded": EXCLUDED,
        "main": {},
        "hard": {},
        "diagnostic": {},
        "group_metrics": {},
    }
    group_root = run_dir / "groups" / label
    for group_dir in sorted(group_root.glob("*")):
        metrics_path = group_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        result["group_metrics"][group_dir.name] = str(metrics_path)
        for suite in ["main", "hard", "diagnostic"]:
            for task, value in (metrics.get(suite) or {}).items():
                if isinstance(value, dict):
                    result[suite][task] = value
        if metrics.get("suite_wall_seconds"):
            result.setdefault("suite_wall_seconds", {}).update(metrics["suite_wall_seconds"])
    return result


def selected(metrics: dict[str, Any], suite: str, tasks: list[str]) -> list[float]:
    out: list[float] = []
    for task in tasks:
        value = metric_value(metrics.get(suite, {}).get(task))
        if value is not None:
            out.append(value)
    return out


def recompute(metrics: dict[str, Any]) -> None:
    main = selected(metrics, "main", MAIN_TASKS)
    hard = selected(metrics, "hard", HARD_TASKS)
    diagnostic = selected(metrics, "diagnostic", DIAGNOSTIC_TASKS)
    overall = main + hard + diagnostic
    if main:
        metrics["main"]["main_mean"] = mean(main)
    if hard:
        metrics["hard"]["hard_mean"] = mean(hard)
    if diagnostic:
        metrics["diagnostic"]["diagnostic_mean"] = mean(diagnostic)
    if overall:
        metrics["scoreable_overall_mean"] = mean(overall)


def write_csv(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    long_path = run_dir / "scores_long.csv"
    wide_path = run_dir / "scores_wide.csv"
    with long_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "suite", "task", "score", "score_percent", "n", "correct"])
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            for suite, tasks in [("main", MAIN_TASKS), ("hard", HARD_TASKS), ("diagnostic", DIAGNOSTIC_TASKS)]:
                for task in tasks:
                    metric = metrics.get(suite, {}).get(task)
                    value = metric_value(metric)
                    writer.writerow(
                        {
                            "model": row["model_name"],
                            "suite": suite,
                            "task": task,
                            "score": "" if value is None else value,
                            "score_percent": "" if value is None else 100.0 * value,
                            "n": metric.get("n", "") if isinstance(metric, dict) else "",
                            "correct": metric.get("correct", "") if isinstance(metric, dict) else "",
                        }
                    )
    with wide_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "overall_mean"] + ALL_TASKS)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            out: dict[str, Any] = {
                "model": row["model_name"],
                "overall_mean": metrics.get("scoreable_overall_mean"),
            }
            for task in ALL_TASKS:
                suite = "main" if task in MAIN_TASKS else "hard" if task in HARD_TASKS else "diagnostic"
                out[task] = metric_value(metrics.get(suite, {}).get(task))
            writer.writerow(out)


def write_summary(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = {"run_dir": str(run_dir), "excluded": EXCLUDED, "models": rows}
    (run_dir / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage 3 DAPO Ablation 16-Arm Evaluation",
        "",
        "Excluded by request: APPS hard, PlanBench, HealthBench, HealthBench Professional.",
        "",
        "| model | overall | main | hard | diagnostic |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {model} | {overall} | {main} | {hard} | {diag} |".format(
                model=row["model_name"],
                overall=pct(metrics.get("scoreable_overall_mean")),
                main=pct(metrics.get("main", {}).get("main_mean")),
                hard=pct(metrics.get("hard", {}).get("hard_mean")),
                diag=pct(metrics.get("diagnostic", {}).get("diagnostic_mean")),
            )
        )
    lines.extend(["", "## Per-Task Metrics", ""])
    for row in rows:
        metrics = row["metrics"]
        lines.append(f"### {row['model_name']}")
        for suite, tasks in [("main", MAIN_TASKS), ("hard", HARD_TASKS), ("diagnostic", DIAGNOSTIC_TASKS)]:
            lines.append("")
            lines.append(f"{suite}:")
            for task in tasks:
                metric = metrics.get(suite, {}).get(task)
                value = metric_value(metric)
                if value is None:
                    lines.append(f"- {task}: NA")
                else:
                    n = metric.get("n", "NA") if isinstance(metric, dict) else "NA"
                    extra = ""
                    if isinstance(metric, dict) and "correct" in metric:
                        extra = f", correct={metric['correct']}"
                    lines.append(f"- {task}: {pct(value)} (n={n}{extra})")
        lines.append("")
    (run_dir / "SUMMARY.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = []
    for model in parse_model_specs(args.model_spec):
        label = model["label"]
        metrics = merge_group_metrics(args.run_dir, label, model)
        recompute(metrics)
        final_dir = args.run_dir / "final" / label
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / "metrics.json"
        final_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append({"model_name": label, "metrics_path": str(final_path), "metrics": metrics})
    write_csv(args.run_dir, rows)
    write_summary(args.run_dir, rows)
    print(json.dumps({"summary": str(args.run_dir / "SUMMARY.md"), "models": [row["model_name"] for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
