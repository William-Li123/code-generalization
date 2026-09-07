#!/usr/bin/env python3
"""Reconstruct the main matched-control support statistics from seed CSVs.

The historical repository did not retain the paper-statistics scripts named in
the manuscript.  This implementation follows the published definitions:

* ``U(c,t) = score(random-drop,t) - score(leave-c-out,t)``;
* seeds are averaged within each backbone before the cross-backbone t-test;
* Benjamini--Hochberg correction is applied jointly to all category/task cells;
* the additive residual is obtained by double-centering the pooled support map;
* permutation nulls re-split three control and three leave-out seed scores
  within every backbone, preserving the backbone strata.
* task-sensitivity profiles average absolute support over categories, and the
  cross-backbone agreement null independently shuffles task labels within each
  backbone while preserving every backbone's marginal profile.

Input files are the wide seed CSVs produced by ``package_main_results.py``.
Scores are expected in percentage points.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


IDENTITY_COLUMNS = {"method", "train_seed", "model", "family", "arm"}
NON_TASK_COLUMNS = {"overall_mean_no_health"}
CONTROL_ARM = "full_sft_30353"
LEAVE_PREFIX = "no_"


@dataclass(frozen=True)
class AnalysisInputs:
    scores: pd.DataFrame
    categories: tuple[str, ...]
    tasks: tuple[str, ...]
    models: tuple[str, ...]
    seeds: tuple[str, ...]


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini--Hochberg adjusted p-values, preserving input order."""

    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    if np.any((values < 0) | (values > 1) | ~np.isfinite(values)):
        raise ValueError("p-values must be finite values in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def additive_residual(matrix: np.ndarray) -> np.ndarray:
    """Return epsilon from U(c,t)=mu+alpha_c+beta_t+epsilon_ct."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("support matrix must be a finite two-dimensional array")
    return (
        values
        - values.mean(axis=1, keepdims=True)
        - values.mean(axis=0, keepdims=True)
        + values.mean()
    )


def load_inputs(paths: list[Path], include_overall: bool = False) -> AnalysisInputs:
    if not paths:
        raise ValueError("At least one seed CSV is required")
    frames = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {"train_seed", "model", "arm"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["_source"] = str(path)
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    scores["train_seed"] = scores["train_seed"].astype(str)

    duplicate = scores.duplicated(["model", "train_seed", "arm"], keep=False)
    if duplicate.any():
        sample = scores.loc[duplicate, ["model", "train_seed", "arm", "_source"]]
        raise ValueError(f"Duplicate model/seed/arm rows:\n{sample.head(10)}")

    arms = set(scores["arm"].astype(str))
    if CONTROL_ARM not in arms:
        raise ValueError(f"Matched control arm {CONTROL_ARM!r} is absent")
    categories = tuple(sorted(arm[len(LEAVE_PREFIX) :] for arm in arms if arm.startswith(LEAVE_PREFIX)))
    if not categories:
        raise ValueError("No leave-category-out arms were found")

    excluded = IDENTITY_COLUMNS | {"_source"}
    if not include_overall:
        excluded |= NON_TASK_COLUMNS
    tasks = tuple(
        column
        for column in scores.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(scores[column])
    )
    if not tasks:
        raise ValueError("No numeric task columns were found")

    models = tuple(sorted(scores["model"].astype(str).unique()))
    seeds = tuple(sorted(scores["train_seed"].unique()))
    expected = {(model, seed) for model in models for seed in seeds}
    for arm in (CONTROL_ARM, *(f"{LEAVE_PREFIX}{category}" for category in categories)):
        present = set(
            scores.loc[scores["arm"] == arm, ["model", "train_seed"]]
            .itertuples(index=False, name=None)
        )
        if present != expected:
            missing_pairs = sorted(expected - present)
            raise ValueError(f"Arm {arm!r} is incomplete; missing {missing_pairs[:8]}")

    return AnalysisInputs(scores, categories, tasks, models, seeds)


def support_long(inputs: AnalysisInputs) -> pd.DataFrame:
    index = ["model", "train_seed"]
    control = inputs.scores.loc[inputs.scores["arm"] == CONTROL_ARM].set_index(index)
    records: list[dict[str, object]] = []
    for category in inputs.categories:
        arm = f"{LEAVE_PREFIX}{category}"
        leave = inputs.scores.loc[inputs.scores["arm"] == arm].set_index(index)
        for model in inputs.models:
            for seed in inputs.seeds:
                key = (model, seed)
                for task in inputs.tasks:
                    control_score = float(control.loc[key, task])
                    leave_score = float(leave.loc[key, task])
                    records.append(
                        {
                            "model": model,
                            "train_seed": seed,
                            "category": category,
                            "task": task,
                            "control_score": control_score,
                            "leave_score": leave_score,
                            "support": control_score - leave_score,
                        }
                    )
    return pd.DataFrame.from_records(records)


def observed_statistics(support: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_backbone = (
        support.groupby(["model", "category", "task"], as_index=False)["support"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "support_mean", "std": "support_sd", "count": "n_seed"})
    )
    per_backbone["support_sem"] = per_backbone["support_sd"] / np.sqrt(per_backbone["n_seed"])
    per_backbone["seed_consistent"] = (
        per_backbone["support_mean"].abs().ge(1.0)
        & per_backbone["support_mean"].abs().ge(2.0 * per_backbone["support_sem"])
    )

    rows: list[dict[str, object]] = []
    for (category, task), group in per_backbone.groupby(["category", "task"], sort=True):
        values = group["support_mean"].to_numpy(dtype=float)
        if np.allclose(values, values[0], rtol=0.0, atol=1e-12):
            if np.isclose(values[0], 0.0, rtol=0.0, atol=1e-12):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = float(np.sign(values[0]) * np.inf), 0.0
        else:
            test = stats.ttest_1samp(values, popmean=0.0)
            statistic, p_value = float(test.statistic), float(test.pvalue)
        pooled = float(values.mean())
        rows.append(
            {
                "category": category,
                "task": task,
                "support_mean": pooled,
                "support_sd_across_backbones": float(values.std(ddof=1)),
                "n_backbone": int(len(values)),
                "positive_backbones": int((values > 0).sum()),
                "sign_agreement": int(max((values > 0).sum(), (values < 0).sum())),
                "t_statistic": statistic,
                "p_value": p_value,
            }
        )
    cells = pd.DataFrame(rows)
    cells["q_value_bh"] = bh_adjust(cells["p_value"])
    cells["fdr_0_10"] = cells["q_value_bh"] <= 0.10
    return per_backbone, cells


def matrix_from_cells(
    cells: pd.DataFrame, categories: tuple[str, ...], tasks: tuple[str, ...], value: str
) -> np.ndarray:
    pivot = cells.pivot(index="category", columns="task", values=value)
    return pivot.loc[list(categories), list(tasks)].to_numpy(dtype=float)


def _cell_score_arrays(inputs: AnalysisInputs) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    scores = inputs.scores.set_index(["model", "train_seed", "arm"])
    arrays: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for model_index, model in enumerate(inputs.models):
        for category_index, category in enumerate(inputs.categories):
            for task_index, task in enumerate(inputs.tasks):
                control = np.array(
                    [float(scores.loc[(model, seed, CONTROL_ARM), task]) for seed in inputs.seeds]
                )
                leave = np.array(
                    [
                        float(scores.loc[(model, seed, f"{LEAVE_PREFIX}{category}"), task])
                        for seed in inputs.seeds
                    ]
                )
                arrays[(model_index, category_index, task_index)] = (control, leave)
    return arrays


def permutation_null(
    inputs: AnalysisInputs,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and additive-residual null maps with shape draws x C x T."""

    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(seed)
    arrays = _cell_score_arrays(inputs)
    raw = np.empty((draws, len(inputs.categories), len(inputs.tasks)), dtype=float)
    residual = np.empty_like(raw)
    n_seed = len(inputs.seeds)
    for draw in range(draws):
        by_backbone = np.empty(
            (len(inputs.models), len(inputs.categories), len(inputs.tasks)), dtype=float
        )
        for key, (control, leave) in arrays.items():
            pooled = np.concatenate([control, leave])
            shuffled = rng.permutation(pooled)
            model_index, category_index, task_index = key
            by_backbone[model_index, category_index, task_index] = (
                shuffled[:n_seed].mean() - shuffled[n_seed:].mean()
            )
        raw[draw] = by_backbone.mean(axis=0)
        residual[draw] = additive_residual(raw[draw])
    return raw, residual


def add_residual_and_null(
    cells: pd.DataFrame,
    inputs: AnalysisInputs,
    raw_null: np.ndarray,
    residual_null: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    out = cells.copy()
    observed = matrix_from_cells(out, inputs.categories, inputs.tasks, "support_mean")
    residual = additive_residual(observed)
    lookup = {
        (category, task): residual[category_index, task_index]
        for category_index, category in enumerate(inputs.categories)
        for task_index, task in enumerate(inputs.tasks)
    }
    out["additive_residual"] = [lookup[(row.category, row.task)] for row in out.itertuples()]

    raw_abs = np.abs(raw_null)
    residual_abs = np.abs(residual_null)
    raw_p: list[float] = []
    residual_p: list[float] = []
    for row in out.itertuples():
        category_index = inputs.categories.index(row.category)
        task_index = inputs.tasks.index(row.task)
        raw_distribution = raw_abs[:, category_index, task_index]
        residual_distribution = residual_abs[:, category_index, task_index]
        raw_p.append(float((np.count_nonzero(raw_distribution >= abs(row.support_mean)) + 1) / (len(raw_distribution) + 1)))
        residual_p.append(float((np.count_nonzero(residual_distribution >= abs(row.additive_residual)) + 1) / (len(residual_distribution) + 1)))
    out["raw_null_p"] = raw_p
    out["residual_null_p"] = residual_p

    summary: dict[str, float | int] = {
        "draws": int(raw_null.shape[0]),
        "observed_mean_abs_support": float(np.abs(observed).mean()),
        "observed_max_abs_support": float(np.abs(observed).max()),
        "raw_null_global_q95": float(np.quantile(raw_abs, 0.95)),
        "observed_fraction_above_raw_global_q95": float(
            (np.abs(observed) > np.quantile(raw_abs, 0.95)).mean()
        ),
        "observed_mean_abs_residual": float(np.abs(residual).mean()),
        "observed_max_abs_residual": float(np.abs(residual).max()),
        "residual_null_global_q95": float(np.quantile(residual_abs, 0.95)),
        "fdr_0_10_cells": int(out["fdr_0_10"].sum()),
    }
    return out, summary


def sensitivity_profiles(
    per_backbone: pd.DataFrame,
    inputs: AnalysisInputs,
) -> pd.DataFrame:
    """Return mean absolute category support for every backbone/task pair."""

    profiles = (
        per_backbone.groupby(["model", "task"], as_index=False)["support_mean"]
        .apply(lambda values: float(np.abs(values).mean()))
        .rename(columns={"support_mean": "mean_abs_support"})
    )
    expected = {(model, task) for model in inputs.models for task in inputs.tasks}
    present = set(profiles[["model", "task"]].itertuples(index=False, name=None))
    if present != expected:
        raise ValueError(
            "Sensitivity profile is incomplete; missing "
            f"{sorted(expected - present)[:8]}"
        )
    return profiles


def _mean_pairwise_correlation(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("profile matrix must contain at least two models and tasks")
    correlations: list[float] = []
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[0]):
            if np.isclose(values[left].std(), 0.0) or np.isclose(values[right].std(), 0.0):
                correlation = 0.0
            else:
                correlation = float(np.corrcoef(values[left], values[right])[0, 1])
            correlations.append(correlation)
    return float(np.mean(correlations))


def sensitivity_label_shuffle(
    profiles: pd.DataFrame,
    inputs: AnalysisInputs,
    draws: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    """Test shared task ordering by independently permuting labels per model."""

    if draws < 1:
        raise ValueError("label-shuffle draws must be positive")
    matrix = (
        profiles.pivot(index="model", columns="task", values="mean_abs_support")
        .loc[list(inputs.models), list(inputs.tasks)]
        .to_numpy(dtype=float)
    )
    observed = _mean_pairwise_correlation(matrix)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=float)
    for draw in range(draws):
        shuffled = np.vstack([rng.permutation(row) for row in matrix])
        null[draw] = _mean_pairwise_correlation(shuffled)

    leave_one_out: dict[str, float] = {}
    for index, model in enumerate(inputs.models):
        leave_one_out[model] = _mean_pairwise_correlation(
            np.delete(matrix, index, axis=0)
        )
    summary: dict[str, object] = {
        "definition": "mean Pearson r across every backbone pair after averaging |support| over categories",
        "draws": int(draws),
        "seed": int(seed),
        "observed_mean_pairwise_pearson_r": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        "null_q95": float(np.quantile(null, 0.95)),
        "one_sided_p": float((np.count_nonzero(null >= observed) + 1) / (draws + 1)),
        "leave_one_backbone_out_mean_pairwise_r": leave_one_out,
    }
    return summary, null


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Seed CSVs from package_main_results.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--permutation-seed", type=int, default=20260823)
    parser.add_argument("--label-shuffle-draws", type=int, default=20000)
    parser.add_argument("--label-shuffle-seed", type=int, default=20260824)
    parser.add_argument("--include-overall", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = load_inputs([path.resolve() for path in args.inputs], args.include_overall)
    support = support_long(inputs)
    per_backbone, cells = observed_statistics(support)
    raw_null, residual_null = permutation_null(
        inputs, draws=args.draws, seed=args.permutation_seed
    )
    cells, summary = add_residual_and_null(cells, inputs, raw_null, residual_null)
    profiles = sensitivity_profiles(per_backbone, inputs)
    sensitivity_summary, _ = sensitivity_label_shuffle(
        profiles,
        inputs,
        draws=args.label_shuffle_draws,
        seed=args.label_shuffle_seed,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    support.to_csv(output / "support_by_seed.csv", index=False)
    per_backbone.to_csv(output / "support_by_backbone.csv", index=False)
    cells.to_csv(output / "support_cells.csv", index=False)
    profiles.to_csv(output / "sensitivity_profiles.csv", index=False)
    metadata = {
        "definition": "control minus leave-category-out, percentage points",
        "control_arm": CONTROL_ARM,
        "models": list(inputs.models),
        "seeds": list(inputs.seeds),
        "categories": list(inputs.categories),
        "tasks": list(inputs.tasks),
        "permutation_seed": args.permutation_seed,
        "label_shuffle": sensitivity_summary,
        **summary,
    }
    (output / "support_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
