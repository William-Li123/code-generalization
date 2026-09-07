#!/usr/bin/env python3
"""Shared implementation for the reconstructed paper artifact entry points.

The manuscript names six scripts that were not present in either recovered
archive.  The public entry points call this module so every table and figure is
derived from the released seed CSVs with one implementation and fixed seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import support_statistics as ss


SEED_FILENAMES = (
    "sft_lora_seed20260730.csv",
    "sft_lora_seed20260731.csv",
    "sft_lora_seed20260801.csv",
)
AVERAGE_FILENAME = "sft_lora_average.csv"
DISPLAY_TASKS = {
    "arc_challenge": "ARC-C",
    "finqa": "FinQA",
    "humaneval": "HumanEval",
    "legalbench": "LegalBench",
    "math500_medium": "MATH-M",
    "mbpp_plus": "MBPP+",
    "medcalc": "MedCalc",
    "math500_high_level": "MATH-H",
    "gsm8k": "GSM8K",
    "mbpp_simple": "MBPP-S",
    "scienceqa": "ScienceQA",
}
DISPLAY_CATEGORIES = {
    "direct_implementation_and_utilities": "Direct impl.",
    "dynamic_programming": "Dyn. prog.",
    "graph_structures_and_stateful_systems": "Graph/state",
    "greedy_search_and_optimization": "Greedy",
    "hashing_counting_and_sets": "Hash/sets",
    "math_and_number_theory": "Math/NT",
    "range_window_and_matrix_processing": "Range/matrix",
    "sequence_transformations": "Sequence",
    "sorting_and_ordered_processing": "Sorting",
    "string_and_parsing": "String",
}
REASONING_TRANSFER_TASKS = (
    "arc_challenge",
    "finqa",
    "legalbench",
    "medcalc",
    "scienceqa",
)
IN_DOMAIN_CODE_MATH_TASKS = (
    "humaneval",
    "mbpp_plus",
    "mbpp_simple",
    "math500_medium",
    "math500_high_level",
    "gsm8k",
)
PAPER_TASK_ORDER = (
    "humaneval",
    "mbpp_simple",
    "mbpp_plus",
    "math500_medium",
    "math500_high_level",
    "gsm8k",
    "arc_challenge",
    "finqa",
    "legalbench",
    "medcalc",
    "scienceqa",
)


def parse_common(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--package-root",
        type=Path,
        required=True,
        help="Result package containing csv/sft_lora_seed*.csv and the average CSV.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--permutation-seed", type=int, default=20260823)
    parser.add_argument("--label-shuffle-draws", type=int, default=20000)
    parser.add_argument("--label-shuffle-seed", type=int, default=20260824)
    return parser


def seed_paths(package_root: Path) -> list[Path]:
    csv_root = package_root.resolve() / "csv"
    paths = [csv_root / name for name in SEED_FILENAMES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing formal seed CSVs: {missing}")
    return paths


def load_average(package_root: Path) -> pd.DataFrame:
    path = package_root.resolve() / "csv" / AVERAGE_FILENAME
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"model", "arm", "overall_mean_no_health", *DISPLAY_TASKS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if len(frame) != 78:
        raise ValueError(f"{path} has {len(frame)} rows, expected 78")
    return frame


def compute(
    package_root: Path,
    draws: int,
    permutation_seed: int,
    label_shuffle_draws: int,
    label_shuffle_seed: int,
) -> dict[str, object]:
    paths = seed_paths(package_root)
    inputs = ss.load_inputs(paths)
    support = ss.support_long(inputs)
    per_backbone, cells = ss.observed_statistics(support)
    raw_null, residual_null = ss.permutation_null(inputs, draws, permutation_seed)
    cells, summary = ss.add_residual_and_null(cells, inputs, raw_null, residual_null)
    profiles = ss.sensitivity_profiles(per_backbone, inputs)
    sensitivity, label_null = ss.sensitivity_label_shuffle(
        profiles, inputs, label_shuffle_draws, label_shuffle_seed
    )
    return {
        "inputs": inputs,
        "support": support,
        "per_backbone": per_backbone,
        "cells": cells,
        "raw_null": raw_null,
        "residual_null": residual_null,
        "summary": summary,
        "profiles": profiles,
        "sensitivity": sensitivity,
        "label_null": label_null,
        "permutation_seed": permutation_seed,
        "label_shuffle_seed": label_shuffle_seed,
    }


def ensure_output(path: Path) -> Path:
    output = path.resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame[list(columns)].copy()
    headers = [str(column) for column in selected.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def support_matrix(bundle: dict[str, object], column: str) -> np.ndarray:
    inputs = bundle["inputs"]
    assert isinstance(inputs, ss.AnalysisInputs)
    cells = bundle["cells"]
    assert isinstance(cells, pd.DataFrame)
    if set(inputs.tasks) != set(PAPER_TASK_ORDER):
        raise ValueError("formal paper task set differs from PAPER_TASK_ORDER")
    return ss.matrix_from_cells(cells, inputs.categories, PAPER_TASK_ORDER, column)


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    *,
    limit: float | None = None,
    colorbar: bool = True,
    annotation_size: float = 6,
) -> object:
    limit = max(float(np.abs(matrix).max()), 0.1) if limit is None else max(limit, 0.1)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=40, ha="right")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:+.1f}",
                ha="center",
                va="center",
                fontsize=annotation_size,
            )
    if colorbar:
        plt.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label="support (pp)")
    return image


def make_overall_table(package_root: Path, output: Path) -> None:
    average = load_average(package_root)
    family = average[["model", "arm", "overall_mean_no_health"]].copy()
    family["reasoning_transfer"] = average[list(REASONING_TRANSFER_TASKS)].mean(axis=1)
    family["in_domain_code_math"] = average[list(IN_DOMAIN_CODE_MATH_TASKS)].mean(axis=1)
    family.to_csv(output / "overall_table.csv", index=False)
    (output / "overall_table.md").write_text(
        markdown_table(family, family.columns), encoding="utf-8"
    )
    matched = family[family["arm"].isin(["base", "full_sft", "full_sft_30353"])].copy()
    rows = []
    for model, group in matched.groupby("model", sort=False):
        indexed = group.set_index("arm")
        row: dict[str, object] = {"model": model}
        for metric in ("reasoning_transfer", "in_domain_code_math", "overall_mean_no_health"):
            row[f"matched_{metric}_delta_vs_base"] = float(
                indexed.loc["full_sft_30353", metric] - indexed.loc["base", metric]
            )
            row[f"complete_{metric}_delta_vs_base"] = float(
                indexed.loc["full_sft", metric] - indexed.loc["base", metric]
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "family_tradeoff_vs_base.csv", index=False)


def make_support_table(bundle: dict[str, object], output: Path) -> None:
    cells = bundle["cells"]
    assert isinstance(cells, pd.DataFrame)
    cells.to_csv(output / "support_table_all_cells.csv", index=False)
    candidates = cells.assign(abs_residual=cells["additive_residual"].abs())
    # The manuscript's Table 5 is not the three largest residuals globally.
    # It first applies the uncorrected p<.05 screen, then retains |epsilon|>=.25.
    shortlist = candidates[
        (candidates["p_value"] < 0.05) & (candidates["abs_residual"] >= 0.25)
    ].sort_values(["p_value", "abs_residual"], ascending=[True, False])
    absorbed = candidates[
        (candidates["p_value"] < 0.05) & (candidates["abs_residual"] < 0.25)
    ].sort_values("p_value")
    if len(shortlist) != 3:
        raise ValueError(f"formal Table-5 reconstruction expected three cells, got {len(shortlist)}")
    columns = [
        "task",
        "category",
        "support_mean",
        "sign_agreement",
        "p_value",
        "q_value_bh",
        "additive_residual",
        "residual_null_p",
    ]
    shortlist[columns].to_csv(output / "support_table_top_residuals.csv", index=False)
    (output / "support_table_top_residuals.md").write_text(
        markdown_table(shortlist, columns), encoding="utf-8"
    )
    absorbed[columns].to_csv(output / "support_table_bias_absorbed.csv", index=False)
    (output / "support_table_bias_absorbed.md").write_text(
        markdown_table(absorbed, columns), encoding="utf-8"
    )


def make_paper_figures(
    package_root: Path, bundle: dict[str, object], output: Path
) -> None:
    inputs = bundle["inputs"]
    assert isinstance(inputs, ss.AnalysisInputs)
    matrix = support_matrix(bundle, "support_mean")
    fig, ax = plt.subplots(figsize=(15, 7))
    draw_heatmap(
        ax,
        matrix,
        [DISPLAY_CATEGORIES.get(item, item) for item in inputs.categories],
        [DISPLAY_TASKS.get(item, item) for item in PAPER_TASK_ORDER],
        "Main matched-control support map",
    )
    save_figure(fig, output, "support_map")

    per_backbone = bundle["per_backbone"]
    assert isinstance(per_backbone, pd.DataFrame)
    global_limit = max(float(per_backbone["support_mean"].abs().max()), 0.1)
    fig, axes = plt.subplots(3, 2, figsize=(22, 18), sharex=True, sharey=True)
    image = None
    for ax, model in zip(axes.ravel(), inputs.models):
        subset = per_backbone[per_backbone["model"] == model]
        values = ss.matrix_from_cells(
            subset, inputs.categories, PAPER_TASK_ORDER, "support_mean"
        )
        consistent = ss.matrix_from_cells(
            subset, inputs.categories, PAPER_TASK_ORDER, "seed_consistent"
        ).astype(bool)
        image = draw_heatmap(
            ax,
            values,
            [DISPLAY_CATEGORIES.get(item, item) for item in inputs.categories],
            [DISPLAY_TASKS.get(item, item) for item in PAPER_TASK_ORDER],
            f"{model} ({int(consistent.sum())}/110 seed-consistent)",
            limit=global_limit,
            colorbar=False,
            annotation_size=4.5,
        )
        for row, column in np.argwhere(consistent):
            ax.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.2,
                )
            )
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.012, pad=0.025, label="support (pp)")
    fig.suptitle("Category-to-task support by backbone (three-seed mean)", fontsize=16)
    fig.subplots_adjust(left=0.14, right=0.87, bottom=0.09, top=0.94, hspace=0.28)
    save_figure(fig, output, "support_map_by_backbone")

    average = load_average(package_root)
    base = average[average["arm"] == "base"].set_index("model")["overall_mean_no_health"]
    trained = average[average["arm"].isin(["full_sft", "full_sft_30353"])].copy()
    trained["delta"] = trained.apply(
        lambda row: float(row["overall_mean_no_health"] - base.loc[row["model"]]), axis=1
    )
    pivot = trained.pivot(index="model", columns="arm", values="delta")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot))
    ax.bar(x - 0.18, pivot["full_sft"], 0.36, label="full 35,974")
    ax.bar(x + 0.18, pivot["full_sft_30353"], 0.36, label="matched 30,353")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, pivot.index, rotation=30, ha="right")
    ax.set_ylabel("overall delta vs Base (pp)")
    ax.legend()
    ax.set_title("Net effect of code SFT")
    save_figure(fig, output, "net_effect_vs_base")

    rows = []
    for model, group in average.groupby("model", sort=False):
        indexed = group.set_index("arm")
        base_row = indexed.loc["base"]
        matched_row = indexed.loc[ss.CONTROL_ARM]
        rows.append(
            {
                "model": model,
                "in_domain_delta": float(
                    matched_row[list(IN_DOMAIN_CODE_MATH_TASKS)].mean()
                    - base_row[list(IN_DOMAIN_CODE_MATH_TASKS)].mean()
                ),
                "transfer_delta": float(
                    matched_row[list(REASONING_TRANSFER_TASKS)].mean()
                    - base_row[list(REASONING_TRANSFER_TASKS)].mean()
                ),
            }
        )
    tradeoff = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.scatter(tradeoff["transfer_delta"], tradeoff["in_domain_delta"], s=50)
    for row in tradeoff.itertuples():
        ax.annotate(row.model, (row.transfer_delta, row.in_domain_delta), fontsize=8)
    ax.set_xlabel("reasoning-transfer change vs Base (pp)")
    ax.set_ylabel("in-domain code + math change vs Base (pp)")
    ax.set_title("Backbone-dependent redistribution under matched code SFT")
    save_figure(fig, output, "in_domain_vs_transfer_tradeoff")

    seed_frames = []
    for path in seed_paths(package_root):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        seed_frames.append(frame)
    seeded = pd.concat(seed_frames, ignore_index=True)
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    for ax, model in zip(axes.ravel(), inputs.models):
        part = seeded[seeded["model"] == model]
        rows = []
        for seed, group in part.groupby("train_seed", sort=True):
            indexed = group.set_index("arm")
            for task in inputs.tasks:
                rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "delta": float(indexed.loc[ss.CONTROL_ARM, task] - indexed.loc["base", task]),
                    }
                )
        deltas = pd.DataFrame(rows)
        stats_frame = deltas.groupby("task", as_index=False)["delta"].agg(["mean", "std"]).reset_index()
        stats_frame = stats_frame.sort_values("mean")
        y = np.arange(len(stats_frame))
        colors = np.where(stats_frame["mean"].to_numpy() >= 0, "#3a9d5d", "#c94c4c")
        ax.barh(y, stats_frame["mean"], xerr=stats_frame["std"], color=colors, alpha=0.9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y, [DISPLAY_TASKS.get(item, item) for item in stats_frame["task"]])
        ax.set_title(model)
        ax.set_xlabel("matched control − Base (pp; mean ± seed SD)")
    fig.suptitle("Net effect of code SFT versus Base by task", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output, "net_effect_by_task")

    metrics = ["overall_mean_no_health", *PAPER_TASK_ORDER]
    metric_labels = ["Overall", *[DISPLAY_TASKS.get(item, item) for item in PAPER_TASK_ORDER]]
    trained_arms = ["full_sft", ss.CONTROL_ARM, *[f"no_{item}" for item in inputs.categories]]
    arm_labels = [
        "complete corpus",
        "matched control",
        *[f"− {DISPLAY_CATEGORIES.get(item, item)}" for item in inputs.categories],
    ]
    matrices: dict[str, np.ndarray] = {}
    for model in inputs.models:
        indexed = average[average["model"] == model].set_index("arm")
        matrices[model] = (
            indexed.loc[trained_arms, metrics].to_numpy(dtype=float)
            - indexed.loc["base", metrics].to_numpy(dtype=float)
        )
    arm_limit = max(float(np.abs(value).max()) for value in matrices.values())
    fig, axes = plt.subplots(3, 2, figsize=(24, 19), sharex=True, sharey=True)
    image = None
    for ax, model in zip(axes.ravel(), inputs.models):
        image = draw_heatmap(
            ax,
            matrices[model],
            arm_labels,
            metric_labels,
            model,
            limit=arm_limit,
            colorbar=False,
            annotation_size=3.8,
        )
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.012, pad=0.025, label="delta vs Base (pp)")
    fig.suptitle("Per-arm delta versus Base (three-seed mean)", fontsize=16)
    fig.subplots_adjust(left=0.13, right=0.87, bottom=0.08, top=0.94, hspace=0.25)
    save_figure(fig, output, "arm_delta_vs_base_by_backbone")


def _group_profile(bundle: dict[str, object], predicate) -> pd.Series:
    profiles = bundle["profiles"]
    assert isinstance(profiles, pd.DataFrame)
    selected = profiles[profiles["model"].map(predicate)]
    return selected.groupby("task")["mean_abs_support"].mean()


def make_headline_figure(
    package_root: Path, bundle: dict[str, object], output: Path
) -> None:
    inputs = bundle["inputs"]
    assert isinstance(inputs, ss.AnalysisInputs)
    average = load_average(package_root)
    base = average[average["arm"] == "base"].set_index("model")["overall_mean_no_health"]
    category_arms = [f"no_{category}" for category in inputs.categories]
    control = average[average["arm"] == ss.CONTROL_ARM].set_index("model")["overall_mean_no_health"]
    differences = []
    for arm in category_arms:
        leave = average[average["arm"] == arm].set_index("model")["overall_mean_no_health"]
        differences.extend((leave - control).tolist())
    base_gap = (
        average[average["arm"] == ss.CONTROL_ARM]
        .set_index("model")["overall_mean_no_health"]
        - base
    ).to_numpy()

    fig = plt.figure(figsize=(20, 6.5))
    axes = fig.subplots(1, 3)
    axes[0].boxplot([differences, base_gap], labels=["leave-out − control", "matched − Base"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("overall difference (pp)")
    axes[0].set_title("A  Aggregate stability")

    matrix = support_matrix(bundle, "support_mean")
    draw_heatmap(
        axes[1],
        matrix,
        [DISPLAY_CATEGORIES.get(item, item) for item in inputs.categories],
        [DISPLAY_TASKS.get(item, item) for item in PAPER_TASK_ORDER],
        "B  Category-to-task support",
    )

    qwen = _group_profile(bundle, lambda model: str(model).startswith("Qwen"))
    other = _group_profile(bundle, lambda model: not str(model).startswith("Qwen"))
    tasks = list(PAPER_TASK_ORDER)
    x = qwen.loc[tasks].to_numpy()
    y = other.loc[tasks].to_numpy()
    correlation = float(np.corrcoef(x, y)[0, 1])
    axes[2].scatter(x, y)
    for index, task in enumerate(tasks):
        axes[2].annotate(DISPLAY_TASKS.get(task, task), (x[index], y[index]), fontsize=7)
    axes[2].set_xlabel("Qwen-family sensitivity (pp)")
    axes[2].set_ylabel("Llama/Gemma sensitivity (pp)")
    axes[2].set_title(f"C  Cross-group sensitivity (r={correlation:.2f})")
    fig.tight_layout()
    save_figure(fig, output, "headline_figure")


def make_extra_figures(bundle: dict[str, object], output: Path) -> None:
    raw_null = bundle["raw_null"]
    assert isinstance(raw_null, np.ndarray)
    observed = np.abs(support_matrix(bundle, "support_mean")).ravel()
    null = np.abs(raw_null).ravel()
    q95 = float(np.quantile(null, 0.95))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null, bins=60, density=True, histtype="step", linewidth=1.5, label="seed-permutation null")
    ax.hist(observed, bins=25, density=True, alpha=0.45, label="observed cells")
    ax.axvline(q95, linestyle="--", color="black", label=f"null q95={q95:.2f}")
    ax.set_xlabel("absolute pooled support (pp)")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title("Raw-scale category-attribution ceiling")
    save_figure(fig, output, "raw_support_permutation_null")

    inputs = bundle["inputs"]
    assert isinstance(inputs, ss.AnalysisInputs)
    per_backbone = bundle["per_backbone"]
    assert isinstance(per_backbone, pd.DataFrame)
    groups = {
        "Qwen family": lambda name: str(name).startswith("Qwen"),
        "Llama/Gemma": lambda name: not str(name).startswith("Qwen"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    for ax, (label, predicate) in zip(axes, groups.items()):
        subset = per_backbone[per_backbone["model"].map(predicate)]
        cells = subset.groupby(["category", "task"], as_index=False)["support_mean"].mean()
        matrix = ss.matrix_from_cells(cells, inputs.categories, PAPER_TASK_ORDER, "support_mean")
        draw_heatmap(
            ax,
            matrix,
            [DISPLAY_CATEGORIES.get(item, item) for item in inputs.categories],
            [DISPLAY_TASKS.get(item, item) for item in PAPER_TASK_ORDER],
            label,
        )
    fig.tight_layout()
    save_figure(fig, output, "practitioner_group_support_map")


def make_residual_null(bundle: dict[str, object], output: Path) -> None:
    inputs = bundle["inputs"]
    assert isinstance(inputs, ss.AnalysisInputs)
    residual = support_matrix(bundle, "additive_residual")
    fig, ax = plt.subplots(figsize=(15, 7))
    draw_heatmap(
        ax,
        residual,
        [DISPLAY_CATEGORIES.get(item, item) for item in inputs.categories],
        [DISPLAY_TASKS.get(item, item) for item in PAPER_TASK_ORDER],
        "Support after removing shared row/column effects",
    )
    save_figure(fig, output, "additive_residual_map")

    pooled = support_matrix(bundle, "support_mean")
    common_limit = max(float(np.abs(pooled).max()), 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(24, 7), sharex=True, sharey=True)
    image = draw_heatmap(
        axes[0],
        pooled,
        [DISPLAY_CATEGORIES.get(item, item) for item in inputs.categories],
        [DISPLAY_TASKS.get(item, item) for item in PAPER_TASK_ORDER],
        "Pooled support",
        limit=common_limit,
        colorbar=False,
    )
    draw_heatmap(
        axes[1],
        residual,
        [DISPLAY_CATEGORIES.get(item, item) for item in inputs.categories],
        [DISPLAY_TASKS.get(item, item) for item in PAPER_TASK_ORDER],
        "Additive residual",
        limit=common_limit,
        colorbar=False,
    )
    fig.colorbar(image, ax=axes.tolist(), fraction=0.015, pad=0.01, label="support / residual (pp)")
    fig.suptitle("Shared-control column bias and bias-corrected residual", fontsize=15)
    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.18, top=0.88, wspace=0.16)
    save_figure(fig, output, "pooled_support_and_residual")

    residual_null = bundle["residual_null"]
    assert isinstance(residual_null, np.ndarray)
    null = np.abs(residual_null).ravel()
    observed = np.abs(residual).ravel()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null, bins=60, density=True, histtype="step", linewidth=1.5, label="residual null")
    ax.hist(observed, bins=25, density=True, alpha=0.45, label="observed residuals")
    ax.axvline(np.quantile(null, 0.95), color="black", linestyle="--", label="null q95")
    ax.set_xlabel("absolute additive residual (pp)")
    ax.set_ylabel("density")
    ax.legend()
    save_figure(fig, output, "residual_scale_permutation_null")

    label_null = bundle["label_null"]
    sensitivity = bundle["sensitivity"]
    assert isinstance(label_null, np.ndarray) and isinstance(sensitivity, dict)
    observed_r = float(sensitivity["observed_mean_pairwise_pearson_r"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(label_null, bins=60, density=True, alpha=0.7)
    ax.axvline(observed_r, color="red", linewidth=2, label=f"observed r={observed_r:.3f}")
    ax.set_xlabel("mean pairwise Pearson r")
    ax.set_ylabel("density")
    ax.set_title("Independent task-label shuffle null")
    ax.legend()
    save_figure(fig, output, "sensitivity_label_shuffle_null")

    cells = bundle["cells"]
    assert isinstance(cells, pd.DataFrame)
    screened = cells[cells["p_value"] < 0.05].copy()
    retained = screened[screened["additive_residual"].abs() >= 0.25]
    absorbed = screened[screened["additive_residual"].abs() < 0.25]
    summary = {
        "reconstruction_notice": (
            "The manuscript-named scripts were absent from both recovered archives; "
            "this implementation follows the published equations."
        ),
        "permutation_seed": bundle["permutation_seed"],
        "label_shuffle_seed": bundle["label_shuffle_seed"],
        "label_shuffle": sensitivity,
        "uncorrected_p_lt_0_05_cells": int(len(screened)),
        "table5_retained_cells": int(len(retained)),
        "bias_absorbed_cells_reconstructed": int(len(absorbed)),
        "manuscript_table6_reported_cells": 8,
        "table6_count_discrepancy": (
            "The released CSVs also place graph_structures_and_stateful_systems/FinQA "
            "at p=0.049138 with |residual|<0.25, yielding nine absorbed cells rather "
            "than the manuscript's eight."
        ),
        **bundle["summary"],
    }
    (output / "null_statistics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_bundle(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    output = ensure_output(args.output)
    bundle = compute(
        args.package_root,
        args.draws,
        args.permutation_seed,
        args.label_shuffle_draws,
        args.label_shuffle_seed,
    )
    return output, bundle
