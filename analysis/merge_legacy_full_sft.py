#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MAIN_TASKS = [
    "arc_challenge",
    "finqa",
    "healthbench",
    "healthbench_professional",
    "humaneval",
    "legalbench",
    "math500_medium",
    "mbpp_plus",
    "medcalc",
]
HARD_TASKS = ["apps_hard", "math500_high_level"]
DIAGNOSTIC_TASKS = ["gsm8k", "mbpp_simple", "scienceqa"]
HEALTH_TASKS = ["healthbench", "healthbench_professional"]
SCOREABLE_MAIN_NO_HEALTH = [task for task in MAIN_TASKS if task not in HEALTH_TASKS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-spec", action="append", required=True, help="label|checkpoint_path")
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


def selected(values: dict[str, Any], tasks: list[str]) -> list[float]:
    out: list[float] = []
    for task in tasks:
        value = metric_value(values.get(task))
        if value is not None:
            out.append(value)
    return out


def parse_model_specs(specs: list[str]) -> list[dict[str, str]]:
    models = []
    for spec in specs:
        parts = spec.split("|", 1)
        if len(parts) != 2:
            raise SystemExit(f"Bad --model-spec {spec!r}; expected label|checkpoint_path")
        models.append({"label": parts[0], "checkpoint_path": parts[1]})
    return models


def merge_group_metrics(run_dir: Path, model: dict[str, str]) -> dict[str, Any]:
    label = model["label"]
    result: dict[str, Any] = {
        "model_name": label,
        "checkpoint_path": model["checkpoint_path"],
        "adapter_path": None,
        "excluded": {"planbench_hard_all_tasks": "excluded by user request"},
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


def add_health_scores(run_dir: Path, label: str, result: dict[str, Any]) -> None:
    score_dir = run_dir / "healthbench" / "scores"
    skipped_path = run_dir / "healthbench" / "skipped" / f"{label}.json"
    skipped = read_json(skipped_path) if skipped_path.exists() else None
    for dataset in HEALTH_TASKS:
        if skipped:
            result["main"][dataset] = {
                "n": 0,
                "score": None,
                "status": skipped.get("status", "skipped"),
                "reason": skipped.get("reason"),
                "skip_marker": str(skipped_path),
            }
            continue
        path = score_dir / f"{dataset}_{label}_scores.summary.json"
        if not path.exists():
            result["main"][dataset] = {
                "n": 0,
                "score": None,
                "status": "missing_healthbench_score",
                "summary_path": str(path),
            }
            continue
        summary = read_json(path)
        result["main"][dataset] = {
            "n": summary.get("n"),
            "score": summary.get("score"),
            "hit_max_new_tokens": summary.get("hit_max_new_tokens"),
            "usage": summary.get("usage", {}),
            "breakdown": summary.get("breakdown", {}),
            "summary_path": str(path),
        }


def recompute(result: dict[str, Any]) -> None:
    main_no_health = selected(result["main"], SCOREABLE_MAIN_NO_HEALTH)
    hard = selected(result["hard"], HARD_TASKS)
    diagnostic = selected(result["diagnostic"], DIAGNOSTIC_TASKS)
    health = selected(result["main"], HEALTH_TASKS)
    all_no_health = main_no_health + hard + diagnostic
    all_with_health = all_no_health + health
    if main_no_health:
        result["main"]["main_scoreable_mean_no_health"] = mean(main_no_health)
    if health:
        result["main"]["healthbench_mean"] = mean(health)
    if main_no_health or health:
        result["main"]["main_scoreable_mean_with_health"] = mean(main_no_health + health)
    if hard:
        result["hard"]["hard_mean"] = mean(hard)
    if diagnostic:
        result["diagnostic"]["diagnostic_mean"] = mean(diagnostic)
    if all_no_health:
        result["scoreable_overall_mean_no_health"] = mean(all_no_health)
    if all_with_health:
        result["scoreable_overall_mean_with_health"] = mean(all_with_health)


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def write_summary(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = {"run_dir": str(run_dir), "models": rows}
    (run_dir / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Stage 4 Full SFT Evaluation",
        "",
        "PlanBench is excluded. HealthBench scores are from the external rubric grader.",
        "",
        "| model | overall no health | overall with health | main no health | health mean | hard | diagnostic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {model} | {overall_no} | {overall_with} | {main_no} | {health} | {hard} | {diag} |".format(
                model=row["model_name"],
                overall_no=pct(metrics.get("scoreable_overall_mean_no_health")),
                overall_with=pct(metrics.get("scoreable_overall_mean_with_health")),
                main_no=pct(metrics.get("main", {}).get("main_scoreable_mean_no_health")),
                health=pct(metrics.get("main", {}).get("healthbench_mean")),
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
                    continue
                n = metric.get("n", "NA") if isinstance(metric, dict) else "NA"
                extra = ""
                if isinstance(metric, dict) and "correct" in metric:
                    extra = f", correct={metric['correct']}"
                if isinstance(metric, dict) and "hit_max_new_tokens" in metric:
                    extra += f", hit_max={metric['hit_max_new_tokens']}"
                lines.append(f"- {task}: {pct(value)} (n={n}{extra})")
        lines.append("")
    (run_dir / "SUMMARY.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = []
    for model in parse_model_specs(args.model_spec):
        label = model["label"]
        metrics = merge_group_metrics(args.run_dir, model)
        add_health_scores(args.run_dir, label, metrics)
        recompute(metrics)
        final_dir = args.run_dir / "final" / label
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / "metrics.json"
        final_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append({"model_name": label, "metrics_path": str(final_path), "metrics": metrics})
    write_summary(args.run_dir, rows)
    print(json.dumps({"summary": str(args.run_dir / "SUMMARY.md"), "models": [row["model_name"] for row in rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
