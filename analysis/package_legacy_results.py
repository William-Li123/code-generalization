#!/usr/bin/env python3
"""Rebuild paper data tables and figures from the canonical 11-task metrics."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy import stats


MODELS = ("Qwen2.5-7B-Instruct", "Qwen3-8B-Base")
MODEL_SHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen3-8B-Base": "Qwen3-8B"}
MODEL_SLUG = {"Qwen2.5-7B-Instruct": "qwen25", "Qwen3-8B-Base": "qwen3"}
SFT_SEEDS = (20260603, 20260604, 20260605)
SFT_ARMS = (
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
CATEGORIES = (
    "math_number_theory",
    "data_structure",
    "dynamic_programming",
    "greedy_search",
    "implementation_simulation",
    "graph",
    "other_algorithm",
)
CATEGORY_ARM = {category: f"no_{category}" for category in CATEGORIES}
CATEGORY_SHORT = {
    "math_number_theory": "Math / NT",
    "data_structure": "Data struct.",
    "dynamic_programming": "Dyn. prog.",
    "greedy_search": "Greedy",
    "implementation_simulation": "Impl. / sim.",
    "graph": "Graph",
    "other_algorithm": "Other alg.",
}
DAPO_SLUG = {
    "full_dapo_12168": "full",
    "no_math_number_theory": "no_math",
    "no_data_structure": "no_data_structure",
    "no_dynamic_programming": "no_dynamic_programming",
    "no_greedy_search": "no_greedy_search",
    "no_implementation_simulation": "no_implementation_simulation",
    "no_graph": "no_graph",
    "no_other_algorithm": "no_other_algorithm",
}
TASKS = (
    "humaneval",
    "mbpp_simple",
    "mbpp_plus",
    "gsm8k",
    "math500_medium",
    "math500_high_level",
    "finqa",
    "medcalc",
    "arc_challenge",
    "legalbench",
    "scienceqa",
)
TASK_LABEL = {
    "humaneval": "HumanEval",
    "mbpp_simple": "MBPP-Simple",
    "mbpp_plus": "MBPP+",
    "gsm8k": "GSM8K",
    "math500_medium": "MATH-Med",
    "math500_high_level": "MATH-High",
    "finqa": "FinQA",
    "medcalc": "MedCalc",
    "arc_challenge": "ARC-C",
    "legalbench": "LegalBench",
    "scienceqa": "ScienceQA",
}
SUITE = {
    "humaneval": "main",
    "mbpp_plus": "main",
    "math500_medium": "main",
    "finqa": "main",
    "medcalc": "main",
    "arc_challenge": "main",
    "legalbench": "main",
    "math500_high_level": "hard",
    "gsm8k": "diagnostic",
    "mbpp_simple": "diagnostic",
    "scienceqa": "diagnostic",
}
DISPLAY_ORDER = [TASK_LABEL[task] for task in TASKS]

BLUE = "#2867b2"
RED = "#cc3d3d"
TEAL = "#117f73"
ORANGE = "#d67b22"
GRID = "#d9dce1"
MUTED = "#656b73"
ALLOW_LEGACY_GSM8K = False


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 260,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metric: dict[str, Any]) -> float:
    for key in ("accuracy", "pass_at_1", "score"):
        if isinstance(metric.get(key), (int, float)):
            return 100.0 * float(metric[key])
    raise KeyError(metric)


def scores(metrics: dict[str, Any]) -> dict[str, float]:
    output = {}
    for task in TASKS:
        output[TASK_LABEL[task]] = metric_value(metrics[SUITE[task]][task])
    output["Overall"] = float(np.mean([output[TASK_LABEL[task]] for task in TASKS]))
    return output


def base_path(root: Path, model: str) -> Path:
    return (
        root
        / "output"
        / "stage2_taco_sft_clean_prompt_eval"
        / "seed_20260605"
        / f"{model}__base"
        / "metrics.json"
    )


def sft_path(root: Path, seed: int, model: str, arm: str) -> Path:
    return (
        root
        / "output"
        / "stage2_taco_sft_clean_prompt_eval_3seed_sft_lora"
        / f"train_seed_{seed}"
        / f"{model}__{arm}"
        / "metrics.json"
    )


def dapo_ablation_path(root: Path, model: str, arm: str) -> Path:
    prefix = "q25" if model.startswith("Qwen2.5") else "q3"
    slug = f"{prefix}_{DAPO_SLUG[arm]}"
    return (
        root
        / "output"
        / "stage3_dapo_ablation16_eval"
        / "stage3-dapo-ablation16-bestval-eval-noapps-nohealth-20260630-163124"
        / "final"
        / slug
        / "metrics.json"
    )


def dapo_complete_path(root: Path, model: str) -> Path:
    slug = "qwen25_7b_final" if model.startswith("Qwen2.5") else "qwen3_8b_final"
    return (
        root
        / "output"
        / "stage3_dapo_three_model_eval"
        / "stage3-dapo-three-model-eval-test-20260611-201711"
        / "final"
        / slug
        / "metrics.json"
    )


def validate_metrics(path: Path) -> dict[str, Any]:
    data = load_json(path)
    gsm = data.get("diagnostic", {}).get("gsm8k", {})
    if not ALLOW_LEGACY_GSM8K and (gsm.get("n") != 1319 or gsm.get("formal_result") is not True):
        raise ValueError(f"Invalid formal GSM8K result: {path}")
    missing = [task for task in TASKS if task not in data.get(SUITE[task], {})]
    if missing:
        raise ValueError(f"Missing tasks {missing}: {path}")
    return data


def collect(root: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    output.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, Any]] = {}

    def ingest(key: str, path: Path) -> dict[str, Any]:
        data = validate_metrics(path)
        cache[key] = data
        destination = output / "metrics" / f"{key.replace('/', '__')}.json"
        shutil.copy2(path, destination)
        return data

    sft_rows = []
    for seed in SFT_SEEDS:
        for model in MODELS:
            for arm in SFT_ARMS:
                values = scores(ingest(f"sft/{seed}/{model}/{arm}", sft_path(root, seed, model, arm)))
                for dataset, score in values.items():
                    sft_rows.append(
                        {
                            "train_seed": seed,
                            "model": model,
                            "arm": arm,
                            "display": f"{MODEL_SHORT[model]} {arm}",
                            "dataset": dataset,
                            "score": score,
                        }
                    )

    dapo_rows = []
    for model in MODELS:
        values = scores(ingest(f"base/{model}", base_path(root, model)))
        for dataset, score in values.items():
            dapo_rows.append({"model": model, "arm": "base", "dataset": dataset, "score": score})
        values = scores(ingest(f"dapo/{model}/full_dapo", dapo_complete_path(root, model)))
        for dataset, score in values.items():
            dapo_rows.append({"model": model, "arm": "full_dapo", "dataset": dataset, "score": score})
        for arm in DAPO_SLUG:
            values = scores(ingest(f"dapo/{model}/{arm}", dapo_ablation_path(root, model, arm)))
            for dataset, score in values.items():
                dapo_rows.append({"model": model, "arm": arm, "dataset": dataset, "score": score})
    return pd.DataFrame(sft_rows), pd.DataFrame(dapo_rows), cache


def bh_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def support_tables(sft: pd.DataFrame, dapo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keyed = sft.set_index(["train_seed", "model", "arm", "dataset"])["score"]
    sft_support_rows = []
    for seed in SFT_SEEDS:
        for model in MODELS:
            for category in CATEGORIES:
                arm = CATEGORY_ARM[category]
                for dataset in DISPLAY_ORDER:
                    support = keyed.loc[(seed, model, "full_sft_14043", dataset)] - keyed.loc[(seed, model, arm, dataset)]
                    sft_support_rows.append(
                        {"seed": seed, "model": model, "category": category, "dataset": dataset, "support": support}
                    )
    sft_support = pd.DataFrame(sft_support_rows)

    summary_rows = []
    for (model, category, dataset), group in sft_support.groupby(["model", "category", "dataset"]):
        values = group.support.to_numpy(float)
        mean = float(values.mean())
        sem = float(stats.sem(values))
        p = float(stats.ttest_1samp(values, 0.0).pvalue)
        summary_rows.append(
            {
                "model": model,
                "category": category,
                "dataset": dataset,
                "mean": mean,
                "std": float(values.std(ddof=1)),
                "sem": sem,
                "p": p,
                "seed_consistent": bool(abs(mean) >= 1.0 and abs(mean) >= 2.0 * sem),
            }
        )
    sft_summary = pd.DataFrame(summary_rows)
    sft_summary["p_bh_77"] = np.nan
    per_backbone = {}
    for model, indices in sft_summary.groupby("model").groups.items():
        indices = list(indices)
        sft_summary.loc[indices, "p_bh_77"] = bh_adjust(sft_summary.loc[indices, "p"].tolist())
        per_backbone[model] = {
            "seed_consistent": int(sft_summary.loc[indices, "seed_consistent"].sum()),
            "uncorrected_p_lt_0_05": int((sft_summary.loc[indices, "p"] < 0.05).sum()),
            "bh_q_0_10": int((sft_summary.loc[indices, "p_bh_77"] <= 0.10).sum()),
        }

    candidate_rows = []
    for (category, dataset), group in sft_summary.groupby(["category", "dataset"]):
        by_model = group.set_index("model")
        means = [float(by_model.loc[model, "mean"]) for model in MODELS]
        if not bool(group.seed_consistent.any()) or np.sign(means[0]) != np.sign(means[1]):
            continue
        values = sft_support[(sft_support.category == category) & (sft_support.dataset == dataset)].support.to_numpy(float)
        candidate_rows.append(
            {
                "category": category,
                "dataset": dataset,
                "qwen25_mean": means[0],
                "qwen3_mean": means[1],
                "pooled_mean": float(values.mean()),
                "positive_runs": int((values > 0).sum()),
                "negative_runs": int((values < 0).sum()),
                "p": float(stats.ttest_1samp(values, 0.0).pvalue),
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values("p").reset_index(drop=True)
        candidates["p_bh"] = bh_adjust(candidates.p.tolist())
        candidates["survives_bh_0_10"] = candidates.p_bh <= 0.10

    dapo_keyed = dapo.set_index(["model", "arm", "dataset"])["score"]
    dapo_rows = []
    for model in MODELS:
        for category in CATEGORIES:
            arm = CATEGORY_ARM[category]
            for dataset in DISPLAY_ORDER:
                support = dapo_keyed.loc[(model, "full_dapo_12168", dataset)] - dapo_keyed.loc[(model, arm, dataset)]
                dapo_rows.append({"model": model, "category": category, "dataset": dataset, "support": support})
    dapo_support = pd.DataFrame(dapo_rows)

    correlations = {}
    for model in MODELS:
        # Sensitivity is mean absolute category support after seed averaging.
        s = (
            sft_summary[sft_summary.model == model]
            .groupby("dataset")
            ["mean"]
            .apply(lambda values: float(np.mean(np.abs(values))))
        )
        d = (
            dapo_support[dapo_support.model == model]
            .groupby("dataset")
            .support.apply(lambda values: float(np.mean(np.abs(values))))
        )
        common = list(DISPLAY_ORDER)
        pearson = stats.pearsonr(s.loc[common], d.loc[common])
        spearman = stats.spearmanr(s.loc[common], d.loc[common])
        correlations[model] = {
            "task_count": len(common),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "sft_sensitivity": {task: float(s.loc[task]) for task in common},
            "dapo_sensitivity": {task: float(d.loc[task]) for task in common},
        }

    statistics = {
        "per_backbone_filter_counts": per_backbone,
        "candidate_count": len(candidates),
        "pooled_candidates": candidates.to_dict(orient="records"),
        "pooled_confirmed_count": int(candidates.survives_bh_0_10.sum()) if not candidates.empty else 0,
        "correlations": correlations,
    }
    return sft_support, sft_summary, dapo_support, statistics


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_tables(output: Path, sft: pd.DataFrame, dapo: pd.DataFrame, sft_support: pd.DataFrame, sft_summary: pd.DataFrame, dapo_support: pd.DataFrame, statistics: dict[str, Any]) -> None:
    data_dir = output / "data"
    save_frame(sft, data_dir / "three_seed_scores_long.csv")
    save_frame(
        sft.groupby(["model", "arm", "display", "dataset"], as_index=False).score.agg(mean_score="mean", std_score="std"),
        data_dir / "three_seed_scores_mean.csv",
    )
    save_frame(
        sft.pivot(index=["train_seed", "model", "arm", "display"], columns="dataset", values="score").reset_index(),
        data_dir / "three_seed_scores_wide_by_seed.csv",
    )
    save_frame(dapo, data_dir / "dapo_scores_long.csv")
    save_frame(sft_support, data_dir / "sft_support_by_seed.csv")
    save_frame(sft_summary, data_dir / "sft_support_summary.csv")
    save_frame(dapo_support, data_dir / "dapo_support.csv")
    (data_dir / "statistical_summary.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def heatmap(ax, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], limit: float, boxes: set[tuple[int, int]] | None = None, daggers: set[tuple[int, int]] | None = None) -> None:
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    boxes = boxes or set()
    daggers = daggers or set()
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            suffix = "†" if (row, col) in daggers else ""
            color = "white" if abs(value) > limit * 0.58 else "#17212b"
            weight = "bold" if (row, col) in boxes or (row, col) in daggers else "normal"
            ax.text(col, row, f"{value:+.1f}{suffix}", ha="center", va="center", fontsize=8, color=color, fontweight=weight)
            if (row, col) in boxes:
                ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.5))
    for boundary in (2.5, 5.5, 7.5):
        ax.axvline(boundary, color="#343a40", linewidth=0.9, alpha=0.75)
    return image


def plot_support_maps(output: Path, sft_summary: pd.DataFrame, dapo_support: pd.DataFrame, statistics: dict[str, Any]) -> None:
    figures = output / "figures"
    confirmed = {
        (row["category"], row["dataset"])
        for row in statistics["pooled_candidates"]
        if row.get("survives_bh_0_10")
    }
    limit = max(7.0, float(sft_summary["mean"].abs().max()), float(dapo_support.support.abs().max()))
    fig, axes = plt.subplots(1, 2, figsize=(17.2, 6.0), sharey=True)
    images = []
    for ax, model in zip(axes, MODELS):
        subset = sft_summary[sft_summary.model == model].set_index(["category", "dataset"])
        matrix = np.array([[subset.loc[(cat, task), "mean"] for task in DISPLAY_ORDER] for cat in CATEGORIES])
        boxes = {
            (ri, ci)
            for ri, cat in enumerate(CATEGORIES)
            for ci, task in enumerate(DISPLAY_ORDER)
            if bool(subset.loc[(cat, task), "seed_consistent"])
        }
        daggers = {(ri, ci) for ri, cat in enumerate(CATEGORIES) for ci, task in enumerate(DISPLAY_ORDER) if (cat, task) in confirmed}
        images.append(heatmap(ax, matrix, [CATEGORY_SHORT[c] for c in CATEGORIES], DISPLAY_ORDER, limit, boxes, daggers))
        count = int(subset.seed_consistent.sum())
        ax.set_title(f"{model}   ({count}/77 seed-consistent)")
    fig.suptitle("Category-to-task support map under LoRA SFT   (boxed = seed-consistent; † = pooled FDR)", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.89, bottom=0.23, top=0.82, wspace=0.20)
    colorbar = fig.colorbar(images[-1], cax=fig.add_axes([0.915, 0.20, 0.015, 0.62]))
    colorbar.set_label("support $U_p(c,t)$ (pp): positive = category supports task")
    fig.savefig(figures / "fig_support_map_lora.png")
    fig.savefig(figures / "fig_support_map_lora.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17.2, 6.0), sharey=True)
    images = []
    for ax, model in zip(axes, MODELS):
        subset = dapo_support[dapo_support.model == model].set_index(["category", "dataset"])
        matrix = np.array([[subset.loc[(cat, task), "support"] for task in DISPLAY_ORDER] for cat in CATEGORIES])
        images.append(heatmap(ax, matrix, [CATEGORY_SHORT[c] for c in CATEGORIES], DISPLAY_ORDER, limit))
        ax.set_title(model)
    fig.suptitle("Category-to-task support map under DAPO", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.89, bottom=0.23, top=0.82, wspace=0.20)
    colorbar = fig.colorbar(images[-1], cax=fig.add_axes([0.915, 0.20, 0.015, 0.62]))
    colorbar.set_label("support $U_p(c,t)$ (pp): positive = category supports task")
    fig.savefig(figures / "fig_support_map_dapo.png")
    fig.savefig(figures / "fig_support_map_dapo.pdf")
    plt.close(fig)


def sensitivity_series(sft_summary: pd.DataFrame, dapo_support: pd.DataFrame, model: str) -> tuple[pd.Series, pd.Series]:
    s = (
        sft_summary[sft_summary.model == model]
        .groupby("dataset")
        ["mean"]
        .apply(lambda values: float(np.mean(np.abs(values))))
    )
    d = (
        dapo_support[dapo_support.model == model]
        .groupby("dataset")
        .support.apply(lambda values: float(np.mean(np.abs(values))))
    )
    return s.loc[DISPLAY_ORDER], d.loc[DISPLAY_ORDER]


def plot_agreement(output: Path, sft_summary: pd.DataFrame, dapo_support: pd.DataFrame, statistics: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8))
    for ax, model in zip(axes, MODELS):
        s, d = sensitivity_series(sft_summary, dapo_support, model)
        ax.scatter(s, d, s=34, color="#3a73a8", zorder=3)
        top = set(s.nlargest(3).index) | set(d.nlargest(3).index)
        for task, x, y in zip(DISPLAY_ORDER, s, d):
            if task in top:
                ax.annotate(task, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        upper = max(float(s.max()), float(d.max())) * 1.12
        ax.plot([0, upper], [0, upper], linestyle="--", color="#9aa0a6", linewidth=1)
        ax.set_xlim(0, max(upper, float(s.max()) * 1.08))
        ax.set_ylim(0, max(upper, float(d.max()) * 1.08))
        corr = statistics["correlations"][model]
        ax.set_title(f"{model}   (Pearson r={corr['pearson_r']:.2f}, Spearman ρ={corr['spearman_rho']:.2f})")
        ax.set_xlabel("LoRA SFT task sensitivity (pp)")
        ax.set_ylabel("DAPO task sensitivity (pp)")
        ax.grid(color="#e5e7eb", linewidth=0.8)
    fig.suptitle("The two regimes agree on which tasks are sensitive to code-category composition", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output / "figures" / "fig_regime_agreement.png")
    fig.savefig(output / "figures" / "fig_regime_agreement.pdf")
    plt.close(fig)


def base_scores(root: Path, model: str) -> dict[str, float]:
    return scores(validate_metrics(base_path(root, model)))


def plot_net_effect(
    output: Path, sft: pd.DataFrame, dapo: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    for ax, model in zip(axes, MODELS):
        control = sft[(sft.model == model) & (sft.arm == "full_sft_14043") & (sft.dataset != "Overall")]
        grouped = control.groupby("dataset").score.agg(["mean", "std"])
        base = (
            dapo[(dapo.model == model) & (dapo.arm == "base")]
            .set_index("dataset")
            .score.to_dict()
        )
        delta = grouped["mean"] - pd.Series(base)
        delta = delta.loc[DISPLAY_ORDER].sort_values()
        std = grouped.loc[delta.index, "std"]
        colors = ["#c93a2b" if value < 0 else "#2f7f33" for value in delta]
        ax.barh(delta.index, delta.values, xerr=std.values, color=colors, edgecolor="white", linewidth=0.5, ecolor="#666")
        ax.axvline(0, color="#555", linewidth=1)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(model)
        ax.set_xlabel("Net effect of random-drop code SFT vs base (pp)")
    fig.suptitle("Code SFT shifts the model globally vs base", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output / "figures" / "fig_net_effect_vs_base.png")
    fig.savefig(output / "figures" / "fig_net_effect_vs_base.pdf")
    plt.close(fig)


def arm_heatmaps(output: Path, sft: pd.DataFrame, dapo: pd.DataFrame) -> None:
    package = output / "figures" / "figs_package"
    for regime, frame, arm_order in (
        ("sft_lora", sft.groupby(["model", "arm", "dataset"], as_index=False).score.mean(), list(SFT_ARMS)),
        ("dapo", dapo, ["full_dapo", "full_dapo_12168"] + [CATEGORY_ARM[c] for c in CATEGORIES]),
    ):
        for model in MODELS:
            base = dapo[(dapo.model == model) & (dapo.arm == "base")].set_index("dataset").score
            subset = frame[(frame.model == model) & frame.arm.isin(arm_order)]
            pivot = subset.pivot(index="arm", columns="dataset", values="score").reindex(index=arm_order, columns=DISPLAY_ORDER + ["Overall"])
            delta = pivot.subtract(base.reindex(pivot.columns), axis=1)
            slug = MODEL_SLUG[model]
            for name, matrix, cmap, center in (
                ("absolute", pivot, "viridis", None),
                ("delta_vs_base", delta, "RdBu_r", 0.0),
            ):
                fig, ax = plt.subplots(figsize=(10.8, 5.1))
                if center is None:
                    image = ax.imshow(matrix.values, cmap=cmap, aspect="auto", vmin=0, vmax=100)
                else:
                    limit = max(1.0, float(np.nanmax(np.abs(matrix.values))))
                    image = ax.imshow(matrix.values, cmap=cmap, aspect="auto", vmin=-limit, vmax=limit)
                ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(matrix.index)), matrix.index)
                ax.set_title(f"{model} - {'LoRA SFT' if regime == 'sft_lora' else 'DAPO'} {name.replace('_', ' ')}")
                fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="score (%)" if name == "absolute" else "delta vs base (pp)")
                fig.tight_layout()
                destination = package / regime / f"{slug}_{name}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(destination)
                plt.close(fig)


def draw_panel_header(ax, letter: str, title: str, color: str) -> None:
    ax.text(0.01, 1.06, letter, transform=ax.transAxes, color="white", fontsize=12, fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.28", facecolor=color, edgecolor="none"))
    ax.text(0.08, 1.06, title, transform=ax.transAxes, color=color, fontsize=13, fontweight="bold", ha="left", va="center")


def plot_headline(output: Path, sft: pd.DataFrame, dapo: pd.DataFrame, sft_summary: pd.DataFrame, dapo_support: pd.DataFrame, statistics: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(16, 4.5))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.18, 1.18, 1.0], wspace=0.38)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(grid[0, index]) for index in range(4)]

    # Panel A: every arm relative to its matched random-drop control.
    y_positions = [3.2, 2.25, 1.15, 0.2]
    row_specs = [(MODELS[0], "sft"), (MODELS[1], "sft"), (MODELS[0], "dapo"), (MODELS[1], "dapo")]
    for y, (model, regime) in zip(y_positions, row_specs):
        if regime == "sft":
            means = sft[(sft.model == model) & (sft.dataset == "Overall")].groupby("arm").score.mean()
            control = means["full_sft_14043"]
            loo = np.array([means[CATEGORY_ARM[category]] - control for category in CATEGORIES])
            complete = means["full_sft"] - control
        else:
            means = dapo[(dapo.model == model) & (dapo.dataset == "Overall")].set_index("arm").score
            control = means["full_dapo_12168"]
            loo = np.array([means[CATEGORY_ARM[category]] - control for category in CATEGORIES])
            complete = means["full_dapo"] - control
        base = float(dapo[(dapo.model == model) & (dapo.arm == "base") & (dapo.dataset == "Overall")].score.iloc[0]) - control
        jitter = np.linspace(-0.16, 0.16, len(loo))
        ax_a.scatter(loo, y + jitter, s=16, color=BLUE, alpha=0.78)
        ax_a.scatter([complete], [y], marker="D", s=38, color="#79a5d2", edgecolor="white", linewidth=0.5)
        ax_a.scatter([base], [y], marker="*", s=80, color="#3f3f3f")
        ax_a.text(-0.03, y, f"{MODEL_SHORT[model]}\n{'LoRA SFT' if regime == 'sft' else 'DAPO RL'}", transform=ax_a.get_yaxis_transform(), ha="right", va="center", fontsize=8)
    ax_a.axvline(0, color="#b8b8b8", linewidth=1)
    ax_a.set_yticks([])
    ax_a.set_xlabel("Δ overall vs control (pp)")
    ax_a.grid(axis="x", color="#ececec")
    ax_a.spines[["top", "right", "left"]].set_visible(False)
    ax_a.scatter([], [], s=16, color=BLUE, label="leave-1-out")
    ax_a.scatter([], [], marker="D", s=30, color="#79a5d2", label="complete")
    ax_a.scatter([], [], marker="*", s=50, color="#3f3f3f", label="base")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8, fontsize=7)
    draw_panel_header(ax_a, "A", "Stable across arms", BLUE)

    # Panel B: show only task rows that contain confirmed or strongest cells.
    pooled_sft = sft_summary.groupby(["category", "dataset"]).mean(numeric_only=True)["mean"]
    confirmed_rows = [row for row in statistics["pooled_candidates"] if row.get("survives_bh_0_10")]
    selected_tasks = []
    for row in confirmed_rows:
        if row["dataset"] not in selected_tasks:
            selected_tasks.append(row["dataset"])
    task_strength = (
        pooled_sft.abs().groupby(level="dataset").max().sort_values(ascending=False)
    )
    for task in task_strength.index:
        if task not in selected_tasks:
            selected_tasks.append(task)
        if len(selected_tasks) >= 5:
            break
    selected_tasks = selected_tasks[:5]
    matrix_sft = np.array([[pooled_sft.loc[(category, task)] for category in CATEGORIES] for task in selected_tasks])
    pooled_dapo = dapo_support.groupby(["category", "dataset"]).support.mean()
    matrix_dapo = np.array([[pooled_dapo.loc[(category, task)] for category in CATEGORIES] for task in selected_tasks])
    combined = np.vstack([matrix_sft, np.full((1, len(CATEGORIES)), np.nan), matrix_dapo])
    limit = max(2.5, float(np.nanmax(np.abs(combined))))
    ax_b.imshow(combined, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax_b.set_xticks(range(len(CATEGORIES)), [CATEGORY_SHORT[c].replace(" ", "\n", 1) for c in CATEGORIES], rotation=35, ha="right", fontsize=6.5)
    ylabels = selected_tasks + [""] + selected_tasks
    ax_b.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    confirmed = {(row["category"], row["dataset"]) for row in confirmed_rows}
    for ri, task in enumerate(selected_tasks):
        for ci, category in enumerate(CATEGORIES):
            value = matrix_sft[ri, ci]
            if abs(value) >= 1.25 or (category, task) in confirmed:
                ax_b.text(ci, ri, f"{value:.1f}", ha="center", va="center", fontsize=7, fontweight="bold" if (category, task) in confirmed else "normal", color="white" if abs(value) > limit * 0.55 else "black")
            if (category, task) in confirmed:
                ax_b.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.2))
            dvalue = matrix_dapo[ri, ci]
            dri = ri + len(selected_tasks) + 1
            if abs(dvalue) >= 2.0:
                ax_b.text(ci, dri, f"{dvalue:.1f}", ha="center", va="center", fontsize=7, fontweight="bold", color="white" if abs(dvalue) > limit * 0.55 else "black")
    ax_b.text(-0.02, 0.99, "LoRA SFT (pooled)", transform=ax_b.transAxes, fontsize=8, color=MUTED, fontstyle="italic", va="top")
    ax_b.text(-0.02, 0.44, "DAPO RL", transform=ax_b.transAxes, fontsize=8, color=MUTED, fontstyle="italic")
    ax_b.tick_params(length=0)
    draw_panel_header(ax_b, "B", "Transfer is sparse", RED)

    # Panel C: compact logic-plausible flow from confirmed cells.
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.axis("off")
    flow_rows = sorted(confirmed_rows, key=lambda row: abs(row["pooled_mean"]), reverse=True)[:6]
    target_names = list(dict.fromkeys(row["dataset"] for row in flow_rows))
    target_rank = {name: index for index, name in enumerate(target_names)}
    raw_sources = list(dict.fromkeys(row["category"] for row in flow_rows))
    source_names = sorted(
        raw_sources,
        key=lambda name: np.mean(
            [target_rank[row["dataset"]] for row in flow_rows if row["category"] == name]
        ),
    )
    source_y = {name: 0.86 - index * (0.68 / max(len(source_names) - 1, 1)) for index, name in enumerate(source_names)}
    target_y = {name: 0.86 - index * (0.68 / max(len(target_names) - 1, 1)) for index, name in enumerate(target_names)}
    for name, y in source_y.items():
        ax_c.text(0.02, y, CATEGORY_SHORT[name].replace(".", ""), ha="left", va="center", fontsize=8)
    for name, y in target_y.items():
        ax_c.text(0.98, y, name, ha="right", va="center", fontsize=8)
    for row in flow_rows:
        positive = row["pooled_mean"] >= 0
        arrow = FancyArrowPatch(
            (0.34, source_y[row["category"]]),
            (0.74, target_y[row["dataset"]]),
            connectionstyle="arc3,rad=0.05",
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.1 + 1.8 * abs(row["pooled_mean"]) / max(abs(item["pooled_mean"]) for item in flow_rows),
            color=TEAL if positive else "#d95549",
            alpha=0.72,
        )
        ax_c.add_patch(arrow)
        x = 0.53
        y = (source_y[row["category"]] + target_y[row["dataset"]]) / 2 + 0.02
        ax_c.text(x, y, f"{row['pooled_mean']:+.1f} pp", color=TEAL if positive else "#d95549", fontsize=7, fontweight="bold", ha="center")
    draw_panel_header(ax_c, "C", "A logic-plausible core", TEAL)

    # Panel D: sensitivity recurrence across both backbones.
    markers = [(MODELS[0], "o", BLUE), (MODELS[1], "^", ORANGE)]
    for model, marker, color in markers:
        s, d = sensitivity_series(sft_summary, dapo_support, model)
        corr = statistics["correlations"][model]
        ax_d.scatter(s, d, marker=marker, s=24, color=color, alpha=0.88, label=f"{MODEL_SHORT[model]} (r={corr['pearson_r']:.2f})")
        top = set(s.nlargest(1).index) | set(d.nlargest(1).index)
        for task in top:
            ax_d.annotate(task, (s.loc[task], d.loc[task]), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color="#444")
    upper = max(ax_d.get_xlim()[1], ax_d.get_ylim()[1])
    ax_d.plot([0, upper], [0, upper], linestyle="--", color="#b5b5b5", linewidth=0.8)
    ax_d.set_xlabel("SFT sensitivity (pp)", fontsize=8)
    ax_d.set_ylabel("RL sensitivity (pp)", fontsize=8)
    ax_d.tick_params(labelsize=7)
    ax_d.legend(frameon=False, fontsize=7, loc="upper left")
    ax_d.spines[["top", "right"]].set_visible(False)
    draw_panel_header(ax_d, "D", "Recurs under RL", ORANGE)

    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.20, top=0.83)
    fig.savefig(output / "figures" / "headline_gsm8k_full.png")
    fig.savefig(output / "figures" / "headline_gsm8k_full.pdf")
    plt.close(fig)


def paper_values(
    sft: pd.DataFrame, dapo: pd.DataFrame, statistics: dict[str, Any]
) -> dict[str, Any]:
    values: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "gsm8k_n": 1319, "models": {}, "statistics": statistics}
    for model in MODELS:
        model_out: dict[str, Any] = {}
        sft_overall = sft[(sft.model == model) & (sft.dataset == "Overall")]
        sft_groups = sft_overall.groupby("arm").score.agg(["mean", "std"])
        loo_sft = sft_groups.loc[[CATEGORY_ARM[c] for c in CATEGORIES], "mean"]
        dapo_overall = dapo[(dapo.model == model) & (dapo.dataset == "Overall")].set_index("arm").score
        loo_dapo = dapo_overall.loc[[CATEGORY_ARM[c] for c in CATEGORIES]]
        model_out["sft"] = {
            "base": float(dapo_overall["base"]),
            "complete_mean": float(sft_groups.loc["full_sft", "mean"]),
            "complete_std": float(sft_groups.loc["full_sft", "std"]),
            "random_drop_mean": float(sft_groups.loc["full_sft_14043", "mean"]),
            "random_drop_std": float(sft_groups.loc["full_sft_14043", "std"]),
            "leave_one_out_min": float(loo_sft.min()),
            "leave_one_out_max": float(loo_sft.max()),
        }
        model_out["dapo"] = {
            "base": float(dapo_overall["base"]),
            "complete": float(dapo_overall["full_dapo"]),
            "random_drop": float(dapo_overall["full_dapo_12168"]),
            "leave_one_out_min": float(loo_dapo.min()),
            "leave_one_out_max": float(loo_dapo.max()),
        }
        values["models"][model] = model_out
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--root",
        type=Path,
        help="External legacy experiment root containing the result artifacts.",
    )
    source.add_argument(
        "--reference-package",
        type=Path,
        help="Small aggregate package containing the released legacy CSVs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-legacy-gsm8k-for-smoke", action="store_true")
    return parser.parse_args()


def collect_reference(package: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the released aggregate tables without requiring raw generations."""

    wide_path = package / "three_seed_scores_wide_by_seed.csv"
    dapo_path = package / "dapo_scores_long.csv"
    if not wide_path.is_file() or not dapo_path.is_file():
        raise FileNotFoundError(
            "reference package must contain three_seed_scores_wide_by_seed.csv "
            "and dapo_scores_long.csv"
        )
    wide = pd.read_csv(wide_path)
    id_columns = ["train_seed", "model", "arm", "display"]
    expected_score_columns = DISPLAY_ORDER + ["Overall"]
    if set(wide.columns) != set(id_columns + expected_score_columns):
        raise ValueError("legacy SFT reference table has an unexpected schema")
    if len(wide) != len(MODELS) * len(SFT_ARMS) * len(SFT_SEEDS):
        raise ValueError("legacy SFT reference table must contain 54 rows")
    sft = wide.melt(
        id_vars=id_columns,
        value_vars=expected_score_columns,
        var_name="dataset",
        value_name="score",
    )
    dapo = pd.read_csv(dapo_path)
    if set(dapo.columns) != {"model", "arm", "dataset", "score"}:
        raise ValueError("legacy DAPO reference table has an unexpected schema")
    if len(dapo) != 240:
        raise ValueError("legacy DAPO reference table must contain 240 rows")
    for frame, label in ((sft, "SFT"), (dapo, "DAPO")):
        if not frame.score.between(0.0, 100.0).all():
            raise ValueError(f"legacy {label} reference scores must be percentages")
    return sft, dapo


def main() -> None:
    global ALLOW_LEGACY_GSM8K
    args = parse_args()
    ALLOW_LEGACY_GSM8K = args.allow_legacy_gsm8k_for_smoke
    configure_style()
    root = args.root.resolve() if args.root else None
    package = args.reference_package.resolve() if args.reference_package else None
    if root is not None and not root.is_dir():
        raise FileNotFoundError(f"Legacy experiment root does not exist: {root}")
    if package is not None and not package.is_dir():
        raise FileNotFoundError(f"Legacy reference package does not exist: {package}")
    if package is not None and args.output is None:
        raise ValueError("--output is required with --reference-package")
    output = args.output or root / "graph" / "gsm8k_full_paper_20260713"
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    if package is not None:
        sft, dapo = collect_reference(package)
    else:
        sft, dapo, _ = collect(root, output)
    sft_support, sft_summary, dapo_support, statistics = support_tables(sft, dapo)
    write_tables(output, sft, dapo, sft_support, sft_summary, dapo_support, statistics)
    plot_support_maps(output, sft_summary, dapo_support, statistics)
    plot_agreement(output, sft_summary, dapo_support, statistics)
    plot_net_effect(output, sft, dapo)
    arm_heatmaps(output, sft, dapo)
    plot_headline(output, sft, dapo, sft_summary, dapo_support, statistics)
    values = paper_values(sft, dapo, statistics)
    (output / "paper_values.json").write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Full-GSM8K paper package\n\n"
        "Generated from the canonical 11-task metrics after replacing the legacy GSM8K n=250 subset "
        "with the pinned official n=1,319 test split. APPS-Hard, PlanBench and HealthBench are not "
        "included in tables, figures or aggregate means.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "sft_rows": len(sft), "dapo_rows": len(dapo), "confirmed": statistics["pooled_confirmed_count"]}, indent=2))


if __name__ == "__main__":
    main()
