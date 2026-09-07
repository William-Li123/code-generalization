from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "analysis" / "support_statistics.py"
    spec = importlib.util.spec_from_file_location("support_statistics", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bh_adjust_is_monotone_in_rank() -> None:
    module = load_module()
    p = np.array([0.04, 0.001, 0.02, 0.2])
    adjusted = module.bh_adjust(p)
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert np.all((adjusted >= p) & (adjusted <= 1.0))


def test_additive_residual_has_zero_row_and_column_means() -> None:
    module = load_module()
    matrix = np.array([[1.0, 3.0, 4.0], [2.0, -1.0, 5.0]])
    residual = module.additive_residual(matrix)
    assert np.allclose(residual.mean(axis=0), 0.0)
    assert np.allclose(residual.mean(axis=1), 0.0)


def test_label_shuffle_is_seeded_and_detects_shared_profile() -> None:
    module = load_module()
    models = ("a", "b", "c")
    tasks = ("t1", "t2", "t3", "t4")
    rows = []
    for model_index, model in enumerate(models):
        for task_index, task in enumerate(tasks):
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "mean_abs_support": float(task_index + 0.01 * model_index),
                }
            )
    inputs = module.AnalysisInputs(
        scores=pd.DataFrame(),
        categories=("c1",),
        tasks=tasks,
        models=models,
        seeds=("1", "2", "3"),
    )
    summary_a, null_a = module.sensitivity_label_shuffle(
        pd.DataFrame(rows), inputs, draws=100, seed=17
    )
    summary_b, null_b = module.sensitivity_label_shuffle(
        pd.DataFrame(rows), inputs, draws=100, seed=17
    )
    assert summary_a == summary_b
    assert np.array_equal(null_a, null_b)
    assert summary_a["observed_mean_pairwise_pearson_r"] > 0.99


def test_reference_results_reproduce_published_global_statistics() -> None:
    module = load_module()
    csv_root = ROOT / "reference_results" / "stage01" / "csv"
    inputs = module.load_inputs(
        [
            csv_root / "sft_lora_seed20260730.csv",
            csv_root / "sft_lora_seed20260731.csv",
            csv_root / "sft_lora_seed20260801.csv",
        ]
    )
    support = module.support_long(inputs)
    per_backbone, cells = module.observed_statistics(support)
    observed = module.matrix_from_cells(
        cells, inputs.categories, inputs.tasks, "support_mean"
    )
    profiles = module.sensitivity_profiles(per_backbone, inputs)
    profile_matrix = (
        profiles.pivot(index="model", columns="task", values="mean_abs_support")
        .loc[list(inputs.models), list(inputs.tasks)]
        .to_numpy(dtype=float)
    )
    assert len(inputs.models) == 6
    assert len(inputs.categories) == 10
    assert len(inputs.tasks) == 11
    assert np.isclose(np.abs(observed).mean(), 0.40797880522718033)
    assert np.isclose(np.abs(observed).max(), 1.798941798941801)
    assert np.isclose(module._mean_pairwise_correlation(profile_matrix), 0.5220975311337198)
