#!/usr/bin/env python3
"""Strictly package and plot the archived two-seed Appendix-A full-SFT probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


MODELS = ("Qwen2.5-7B-Instruct", "Qwen3-8B-Base")
SEEDS = (20260603, 20260604)
ARMS = (
    "full_sft",
    "full_sft_14043",
    "no_math_number_theory",
    "no_data_structure",
    "no_dynamic_programming",
    "no_greedy_search",
    "no_implementation_simulation",
    "no_graph",
    "no_other_algorithm",
)
TASKS = (
    "arc_challenge",
    "finqa",
    "humaneval",
    "legalbench",
    "math500_medium",
    "mbpp_plus",
    "medcalc",
    "math500_high_level",
    "mbpp_simple",
    "scienceqa",
)
TASK_N = {
    "humaneval": 164,
    "mbpp_simple": 257,
    "mbpp_plus": 378,
    "math500_medium": 105,
    "math500_high_level": 262,
    "finqa": 1133,
    "medcalc": 1100,
    "arc_challenge": 1172,
    "legalbench": 1689,
    "scienceqa": 2224,
}
SUITE = {
    **{task: "main" for task in ("humaneval", "mbpp_plus", "math500_medium", "finqa", "medcalc", "arc_challenge", "legalbench")},
    "math500_high_level": "hard",
    **{task: "diagnostic" for task in ("mbpp_simple", "scienceqa")},
}
DISPLAY_TASKS = {
    "humaneval": "HumanEval",
    "mbpp_simple": "MBPP-S",
    "mbpp_plus": "MBPP+",
    "math500_medium": "MATH-M",
    "math500_high_level": "MATH-H",
    "finqa": "FinQA",
    "medcalc": "MedCalc",
    "arc_challenge": "ARC-C",
    "legalbench": "Legal",
    "scienceqa": "SciQA",
    "overall": "Overall",
}
METHOD = "full_sft"
FAMILY = {
    "Qwen2.5-7B-Instruct": "qwen25",
    "Qwen3-8B-Base": "qwen3",
}
CSV_FIELDS = (
    "method",
    "train_seed",
    "model",
    "family",
    "arm",
    "overall_mean_no_health",
    *TASKS,
)


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def parse_args() -> argparse.Namespace:
    output_root = env_path("CG_OUTPUT_ROOT")
    default_eval = (
        output_root / "stage06_legacy_full_sft_robustness" / "evaluation" / "appendix_a_10_no_gsm8k"
        if output_root
        else None
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument(
        "--reference-package",
        type=Path,
        help="Released three-CSV Stage-06 package; avoids requiring raw metrics.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.dpi < 80:
        parser.error("--dpi must be at least 80 for readable publication figures")
    if args.eval_root and args.reference_package:
        parser.error("choose exactly one of --eval-root and --reference-package")
    if not args.eval_root and not args.reference_package:
        if default_eval is None:
            parser.error("--eval-root or --reference-package is required")
        args.eval_root = default_eval
    return args


def metric_value(metric: dict[str, Any], path: Path, task: str) -> float:
    for key in ("accuracy", "pass_at_1", "score"):
        value = metric.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{path}: {task} score outside [0,1]: {numeric}")
            return 100.0 * numeric
    raise ValueError(f"{path}: {task} has no finite score")


def read_scores(path: Path, expected_name: str) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("model_name") != expected_name:
        raise ValueError(f"{path}: model_name={data.get('model_name')!r}, expected {expected_name!r}")
    if set(data.get("tasks_requested", [])) != set(TASKS):
        raise ValueError(f"{path}: not the Appendix-A ten-task contract")
    scores: dict[str, float] = {}
    for task in TASKS:
        metric = data.get(SUITE[task], {}).get(task)
        if not isinstance(metric, dict) or metric.get("n") != TASK_N[task]:
            raise ValueError(f"{path}: {task} expected n={TASK_N[task]}, got {metric}")
        scores[task] = metric_value(metric, path, task)
    scores["overall"] = statistics.fmean(scores.values())
    return scores


def collect(
    eval_root: Path,
) -> tuple[dict[str, dict[str, float]], dict[tuple[int, str, str], dict[str, float]]]:
    base: dict[str, dict[str, float]] = {}
    for model in MODELS:
        path = eval_root / "base" / f"{model}__base" / "metrics.json"
        base[model] = read_scores(path, f"{model}__base")
    trained: dict[tuple[int, str, str], dict[str, float]] = {}
    for seed in SEEDS:
        for model in MODELS:
            for arm in ARMS:
                path = eval_root / f"train_seed_{seed}" / f"{model}__{arm}" / "metrics.json"
                trained[(seed, model, arm)] = read_scores(path, f"{model}__{arm}")
    if len(base) != 2 or len(trained) != 36:
        raise RuntimeError(f"expected 2 Base plus 36 trained results, got {len(base)} plus {len(trained)}")
    return base, trained


def collect_reference(
    package: Path,
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Load the released ten-task aggregate tables for analysis reproduction."""

    expected_fields = set(CSV_FIELDS)

    def load(path: Path, expected_seed: int | str) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 20 or any(set(row) != expected_fields for row in rows):
            raise ValueError(f"{path}: expected 20 rows with the Stage-06 schema")
        for row in rows:
            if str(row["train_seed"]) != str(expected_seed):
                raise ValueError(f"{path}: unexpected train_seed={row['train_seed']}")
        validate_rows(rows, str(path))
        return rows

    per_seed = {
        seed: load(package / f"{METHOD}_seed{seed}.csv", seed) for seed in SEEDS
    }
    averaged = load(package / f"{METHOD}_average.csv", "average")
    return per_seed, averaged


def wide_row(train_seed: int | str, model: str, arm: str, scores: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": METHOD,
        "train_seed": train_seed,
        "model": model,
        "family": FAMILY[model],
        "arm": arm,
        "overall_mean_no_health": scores["overall"],
    }
    row.update({task: scores[task] for task in TASKS})
    return row


def seed_rows(
    seed: int,
    base: dict[str, dict[str, float]],
    trained: dict[tuple[int, str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        rows.append(wide_row(seed, model, "base", base[model]))
        rows.extend(wide_row(seed, model, arm, trained[(seed, model, arm)]) for arm in ARMS)
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 rows for seed {seed}, got {len(rows)}")
    return rows


def average_rows(
    base: dict[str, dict[str, float]],
    trained: dict[tuple[int, str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score_names = (*TASKS, "overall")
    for model in MODELS:
        rows.append(wide_row("average", model, "base", base[model]))
        for arm in ARMS:
            means = {
                name: statistics.fmean(trained[(seed, model, arm)][name] for seed in SEEDS)
                for name in score_names
            }
            rows.append(wide_row("average", model, arm, means))
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 average rows, got {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows: list[dict[str, Any]], label: str) -> None:
    expected_arms = ("base", *ARMS)
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if tuple(row["arm"] for row in model_rows) != expected_arms:
            raise ValueError(f"{label}: unstable or incomplete arm order for {model}")
        for row in model_rows:
            for field in ("overall_mean_no_health", *TASKS):
                value = float(row[field])
                if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                    raise ValueError(f"{label}: {model}/{row['arm']}/{field} outside [0,100]")


def annotate(ax: Any, values: Any, *, delta: bool) -> None:
    bound = max(1.0, float(abs(values).max())) if delta else None
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = float(values[row_index, column_index])
            text = f"{value:+.1f}" if delta else f"{value:.1f}"
            if delta:
                color = "white" if abs(value) >= 0.55 * float(bound) else "black"
            else:
                color = "white" if value < 45.0 else "black"
            ax.text(column_index, row_index, text, ha="center", va="center", fontsize=8, color=color)


def save_heatmap(
    values: Any,
    rows: list[str],
    columns: list[str],
    title: str,
    output: Path,
    dpi: int,
    *,
    delta: bool,
) -> None:
    import matplotlib.pyplot as plt

    width = max(12.5, 1.02 * len(columns) + 4.0)
    height = max(6.5, 0.58 * len(rows) + 3.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    if delta:
        bound = max(1.0, float(abs(values).max()))
        image = ax.imshow(values, aspect="auto", vmin=-bound, vmax=bound, cmap="RdBu_r")
        color_label = "Delta vs Base (percentage points)"
    else:
        image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=100.0, cmap="viridis")
        color_label = "Score (%)"
    ax.set_xticks(range(len(columns)), columns, rotation=34, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Arm")
    ax.set_title(title)
    annotate(ax, values, delta=delta)
    fig.colorbar(image, ax=ax, pad=0.015, label=color_label)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np

    value_fields = ("overall_mean_no_health", *TASKS)
    columns = [DISPLAY_TASKS["overall"], *(DISPLAY_TASKS[task] for task in TASKS)]
    outputs: list[Path] = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        absolute = np.asarray([[float(row[field]) for field in value_fields] for row in model_rows], dtype=float)
        base = absolute[0]
        delta = absolute - base
        if not np.allclose(delta[0], 0.0):
            raise ValueError(f"Base delta row is not zero for {model}")
        alias = FAMILY[model]
        absolute_path = output_dir / f"{alias}_absolute.png"
        delta_path = output_dir / f"{alias}_delta_vs_base.png"
        row_labels = [str(row["arm"]) for row in model_rows]
        save_heatmap(
            absolute,
            row_labels,
            columns,
            f"Appendix A full-SFT — {model}: absolute scores",
            absolute_path,
            dpi,
            delta=False,
        )
        save_heatmap(
            delta,
            row_labels,
            columns,
            f"Appendix A full-SFT — {model}: delta vs Base",
            delta_path,
            dpi,
            delta=True,
        )
        outputs.extend((absolute_path, delta_path))
    return outputs


def write_readme(output_dir: Path, source: Path, source_kind: str) -> Path:
    task_counts = ", ".join(f"{task}={TASK_N[task]}" for task in TASKS)
    readme = output_dir / "README.md"
    readme.write_text(
        f"""# Appendix A full-SFT robustness package

- Source: `{source.resolve()}` ({source_kind}). Raw-metric mode expects
  `base/<model>__base/metrics.json` and
  `train_seed_<seed>/<model>__<arm>/metrics.json`; released-package mode uses
  the three hash-frozen aggregate CSVs.
- Coverage: two models, nine trained arms, archived train seeds 20260603/20260604,
  and one Base reference reused in each seed table.
- Evaluation: `appendix_a_10_no_gsm8k`, evaluation seed 20260605, plain prompt mode,
  temperature 0.2, top-p 0.95, and the shared legacy scorer. Code generation uses
  max-new-tokens 1024; other task-specific limits are defined by that evaluator.
- Checkpoints: final exported full-parameter Hugging Face checkpoints. This package
  does not select checkpoints using test metrics.
- Overall: unweighted arithmetic mean of the ten displayed datasets. GSM8K,
  APPS, PlanBench, HealthBench, and HealthBench Professional are excluded.
- Required sample counts: {task_counts}.
- CSVs contain percentages in [0,100], Base first for each model, stable arm order,
  one file per actual seed, and one two-seed average file.
- Figures use the two-seed average. Every delta is `arm - Base`; absolute heatmaps
  use a fixed 0..100 scale and delta heatmaps use symmetric zero-centered limits.
- Comparability caveat: the manuscript reports train seeds 20260604/20260605, but
  recovered execution records identify 20260603/20260604 as train seeds and
  20260605 as evaluation seed. Recovered GSM8K n=250 and APPS diagnostics are
  retained only as archive evidence and do not enter Figure 7.
""",
        encoding="utf-8",
    )
    return readme


def prepare_layout(output_dir: Path) -> tuple[Path, Path]:
    allowed = {"csv", "picture", "README.md"}
    if output_dir.exists():
        extra = sorted(path.name for path in output_dir.iterdir() if path.name not in allowed)
        if extra:
            raise ValueError(f"output package contains unexpected root entries: {extra}")
    csv_dir = output_dir / "csv"
    picture_dir = output_dir / "picture" / METHOD
    csv_dir.mkdir(parents=True, exist_ok=True)
    picture_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, picture_dir


def validate_package(output_dir: Path, csv_paths: list[Path], figure_paths: list[Path]) -> None:
    if {path.name for path in output_dir.iterdir()} != {"csv", "picture", "README.md"}:
        raise ValueError("package root must contain only csv/, picture/, and README.md")
    if len(csv_paths) != 3 or any(not path.is_file() or path.stat().st_size == 0 for path in csv_paths):
        raise ValueError("formal package requires two seed CSVs plus one average CSV")
    if {path.name for path in (output_dir / "csv").iterdir()} != {path.name for path in csv_paths}:
        raise ValueError("csv/ contains unexpected or missing files")
    picture_root = output_dir / "picture"
    if {path.name for path in picture_root.iterdir()} != {METHOD}:
        raise ValueError(f"picture/ must contain only {METHOD}/")
    if len(figure_paths) != 4 or any(not path.is_file() or path.stat().st_size == 0 for path in figure_paths):
        raise ValueError("formal package requires paired absolute/delta PNGs for both models")
    if {path.name for path in (picture_root / METHOD).iterdir()} != {path.name for path in figure_paths}:
        raise ValueError(f"picture/{METHOD}/ contains unexpected or missing files")


def main() -> None:
    args = parse_args()
    csv_dir, picture_dir = prepare_layout(args.output_dir)
    if args.reference_package:
        per_seed, averaged = collect_reference(args.reference_package)
        source = args.reference_package
        source_kind = "released aggregate reference package"
    else:
        base, trained = collect(args.eval_root)
        per_seed = {seed: seed_rows(seed, base, trained) for seed in SEEDS}
        averaged = average_rows(base, trained)
        source = args.eval_root
        source_kind = "38 strict raw metrics"
    for seed, rows in per_seed.items():
        validate_rows(rows, f"seed {seed}")
    validate_rows(averaged, "average")

    csv_paths = [csv_dir / f"{METHOD}_seed{seed}.csv" for seed in SEEDS]
    for seed, path in zip(SEEDS, csv_paths):
        write_csv(path, per_seed[seed])
    average_path = csv_dir / f"{METHOD}_average.csv"
    write_csv(average_path, averaged)
    csv_paths.append(average_path)

    figure_paths = plot(averaged, picture_dir, args.dpi)
    readme = write_readme(args.output_dir, source, source_kind)
    validate_package(args.output_dir, csv_paths, figure_paths)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "input_metrics": 38,
                "csv_files": len(csv_paths),
                "png_files": len(figure_paths),
                "readme": str(readme),
            }
        )
    )


if __name__ == "__main__":
    main()
