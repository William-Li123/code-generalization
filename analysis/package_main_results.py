#!/usr/bin/env python3
"""Build the clean Stage 1 SFT-LoRA result package from formal seed summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(".")
BUILD_ROOT = Path(".")
CSV_DIR = BUILD_ROOT / "csv"
PICTURE_DIR = BUILD_ROOT / "picture" / "sft_lora"

SEEDS = ("20260730", "20260731", "20260801")
MODELS = {
    "Gemma-2-9B-Instruct": "gemma2",
    "Llama-3.1-8B-Instruct": "llama31",
    "Llama-3.2-3B-Instruct": "llama32",
    "Qwen1.5-7B-Chat": "qwen15",
    "Qwen2.5-7B-Instruct": "qwen25",
    "Qwen3-4B-Instruct-2507": "qwen3_4b",
}
EXPECTED_ARMS = (
    "base",
    "full_sft",
    "full_sft_30353",
    "no_direct_implementation_and_utilities",
    "no_dynamic_programming",
    "no_graph_structures_and_stateful_systems",
    "no_greedy_search_and_optimization",
    "no_hashing_counting_and_sets",
    "no_math_and_number_theory",
    "no_range_window_and_matrix_processing",
    "no_sequence_transformations",
    "no_sorting_and_ordered_processing",
    "no_string_and_parsing",
)

DATASETS = (
    "arc_challenge",
    "finqa",
    "humaneval",
    "legalbench",
    "math500_medium",
    "mbpp_plus",
    "medcalc",
    "math500_high_level",
    "gsm8k",
    "mbpp_simple",
    "scienceqa",
)
CSV_COLUMNS = (
    "method",
    "train_seed",
    "model",
    "family",
    "arm",
    "overall_mean_no_health",
    *DATASETS,
)
PLOT_COLUMNS = ("overall_mean_no_health", *DATASETS)
PLOT_LABELS = (
    "Overall",
    "ARC-C",
    "FinQA",
    "HumanEval",
    "LegalBench",
    "MATH Med",
    "MBPP+",
    "MedCalc",
    "MATH High",
    "GSM8K",
    "MBPP Simple",
    "ScienceQA",
)
SOURCE_TO_OUTPUT = {
    "overall": "overall_mean_no_health",
    "arc_challenge": "arc_challenge",
    "finqa": "finqa",
    "humaneval": "humaneval",
    "legalbench": "legalbench",
    "math500_medium": "math500_medium",
    "mbpp_plus": "mbpp_plus",
    "medcalc": "medcalc",
    "math500_high_level": "math500_high_level",
    "gsm8k": "gsm8k",
    "mbpp_simple": "mbpp_simple",
    "scienceqa": "scienceqa",
}


def load_seed(seed: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model, alias in MODELS.items():
        source = SOURCE_ROOT / f"seed_{seed}" / model / "absolute.csv"
        frame = pd.read_csv(source, encoding="utf-8-sig")
        if tuple(frame["checkpoint"]) != EXPECTED_ARMS:
            raise ValueError(f"Unexpected arm order in {source}")
        missing = set(SOURCE_TO_OUTPUT) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {source}: {sorted(missing)}")

        out = pd.DataFrame()
        out["method"] = ["sft_lora"] * len(frame)
        out["train_seed"] = [seed] * len(frame)
        out["model"] = [model] * len(frame)
        out["family"] = [alias] * len(frame)
        out["arm"] = frame["checkpoint"]
        for source_col, output_col in SOURCE_TO_OUTPUT.items():
            values = pd.to_numeric(frame[source_col], errors="raise")
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"Out-of-range score in {source}: {source_col}")
            out[output_col] = values * 100.0

        recomputed = out[list(DATASETS)].mean(axis=1)
        if not np.allclose(
            recomputed,
            out["overall_mean_no_health"],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError(f"Overall mismatch in {source}")
        frames.append(out[list(CSV_COLUMNS)])
    return pd.concat(frames, ignore_index=True)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotate(ax: plt.Axes, values: np.ndarray, delta: bool) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if delta:
                label = f"{value:+.1f}"
                color = "white" if abs(value) >= 0.65 * np.nanmax(np.abs(values)) else "black"
            else:
                label = f"{value:.1f}"
                color = "white" if value < 25 else "black"
            ax.text(col, row, label, ha="center", va="center", fontsize=8.5, color=color)


def plot_model(frame: pd.DataFrame, model: str, alias: str) -> None:
    model_frame = frame[frame["model"] == model].copy()
    model_frame["arm"] = pd.Categorical(
        model_frame["arm"], categories=EXPECTED_ARMS, ordered=True
    )
    model_frame = model_frame.sort_values("arm")
    if tuple(model_frame["arm"].astype("object")) != EXPECTED_ARMS:
        raise ValueError(f"Incomplete averaged arms for {model}")

    absolute = model_frame[list(PLOT_COLUMNS)].to_numpy(dtype=float)
    base = absolute[0]
    delta = absolute - base
    if not np.allclose(delta[0], 0.0, atol=1e-12):
        raise ValueError(f"Non-zero Base delta for {model}")

    row_labels = model_frame["arm"].astype("object").tolist()
    width = 20
    height = 10.8

    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    image = ax.imshow(absolute, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    annotate(ax, absolute, delta=False)
    ax.set_xticks(range(len(PLOT_LABELS)), PLOT_LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Arm")
    ax.set_title(f"SFT LoRA {model}: absolute scores (3-seed mean)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label("Score (%)")
    fig.subplots_adjust(left=0.29, right=0.93, bottom=0.17, top=0.93)
    fig.savefig(PICTURE_DIR / f"{alias}_absolute.png", bbox_inches="tight")
    plt.close(fig)

    limit = max(1.0, float(np.nanmax(np.abs(delta))))
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(delta, cmap="RdBu_r", norm=norm, aspect="auto")
    annotate(ax, delta, delta=True)
    ax.set_xticks(range(len(PLOT_LABELS)), PLOT_LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Arm")
    ax.set_title(f"SFT LoRA {model}: delta vs Base (3-seed mean)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label("Delta vs Base (percentage points)")
    fig.subplots_adjust(left=0.29, right=0.93, bottom=0.17, top=0.93)
    fig.savefig(PICTURE_DIR / f"{alias}_delta_vs_base.png", bbox_inches="tight")
    plt.close(fig)


def write_readme(average: pd.DataFrame) -> None:
    summaries = []
    for model in MODELS:
        part = average[average["model"] == model]
        base = float(part.loc[part["arm"] == "base", "overall_mean_no_health"].iloc[0])
        trained = part[part["arm"] != "base"]
        best = trained.loc[trained["overall_mean_no_health"].idxmax()]
        summaries.append(
            f"| {model} | {base:.2f}% | `{best['arm']}` | "
            f"{best['overall_mean_no_health']:.2f}% | "
            f"{best['overall_mean_no_health'] - base:+.2f} pp |"
        )

    model_list = "\n".join(f"- {model}" for model in MODELS)
    summary_table = "\n".join(summaries)
    readme = f"""# Code Generalization Stage 1 Results

This is the clean publication package for the completed Stage 1 SFT-LoRA
code-category ablation on the new reviewed KodCode-derived V2 data.

## Coverage

- Method: SFT LoRA code-category ablation.
- Training/evaluation seeds: `20260730`, `20260731`, `20260801`.
- Per model and seed: Base plus 12 trained arms.
- Formal checkpoint: final LoRA adapter at 1.0 epoch.
- Every average cell is the arithmetic mean of the three matching formal seed
  results. Base is independently evaluated once per seed.

Included models:

{model_list}

DeepSeek-Coder-6.7B-Instruct is intentionally excluded. Its stored code
generations contain undecoded tokenizer artifacts such as `Ċ` and `Ġ`, making
all three code benchmarks fail parsing. Those measurements are not comparable
model-quality results and are not present in these CSVs or figures.

## Files

```text
csv/
  sft_lora_seed20260730.csv
  sft_lora_seed20260731.csv
  sft_lora_seed20260801.csv
  sft_lora_average.csv
picture/sft_lora/
  <model>_absolute.png
  <model>_delta_vs_base.png
README.md
package_manifest.json
```

Each CSV contains all six models. Each picture shows Base followed by the 12
trained arms for one model. Absolute figures use a fixed `0..100` scale; delta
figures subtract that model's averaged Base row.

## Training contract

- Base weights frozen; only LoRA parameters trained.
- One epoch; learning rate `2e-5`; cosine schedule; warmup ratio `0.03`.
- LoRA rank/alpha/dropout: `16 / 32 / 0.05`.
- Assistant-token-only loss and each tokenizer's native chat template.
- Qwen thinking disabled.
- `full_sft`: 35,974 rows.
- `full_sft_30353`: deterministic 30,353-row equal-size random control.
- Ten `no_*` arms: 30,353 rows each, with the named category fully removed.
- Formal test results were not used to choose checkpoints.

## Evaluation contract

- Native model chat template; Qwen thinking disabled.
- Temperature `0.2`; top-p `0.95`; code max-new-tokens `2048`.
- Full GSM8K official test split (`n=1319`).
- HumanEval `n=164`; MBPP+ `n=378`; MBPP Simple `n=257`.
- Displayed datasets: ARC-Challenge, FinQA, HumanEval, LegalBench,
  MATH-500 Medium, MBPP+, MedCalc, MATH-500 High, GSM8K, MBPP Simple,
  and ScienceQA.
- APPS, PlanBench, HealthBench, and HealthBench Professional are excluded.
- `overall_mean_no_health` is the unweighted mean of the 11 displayed datasets.

## Best Overall arm by model

| Model | Base | Best trained arm | Best Overall | Delta |
|---|---:|---|---:|---:|
{summary_table}

## Provenance

The package is generated from the formal per-seed summaries supplied through
`--source-root`.  The expected source layout is:

```text
<source-root>/seed_<seed>/<model>/absolute.csv
```

Private scheduler identifiers and mount paths are intentionally not embedded
in this public package.
"""
    (BUILD_ROOT / "README.md").write_text(readme, encoding="utf-8")


def validate(seed_frames: dict[str, pd.DataFrame], average: pd.DataFrame) -> None:
    expected_rows = len(MODELS) * len(EXPECTED_ARMS)
    for seed, frame in seed_frames.items():
        if len(frame) != expected_rows:
            raise ValueError(f"Seed {seed} has {len(frame)} rows, expected {expected_rows}")
        if frame.isna().any().any():
            raise ValueError(f"Seed {seed} contains missing values")
    if len(average) != expected_rows or average.isna().any().any():
        raise ValueError("Average CSV is incomplete")
    if average["model"].str.contains("DeepSeek", case=False).any():
        raise ValueError("DeepSeek must not be included")

    for model, alias in MODELS.items():
        for suffix in ("absolute", "delta_vs_base"):
            path = PICTURE_DIR / f"{alias}_{suffix}.png"
            with Image.open(path) as image:
                width, height = image.size
                if width < 2400 or height < 1400:
                    raise ValueError(f"Unexpectedly small figure: {path} {image.size}")


def write_package_manifest(seed_frames: dict[str, pd.DataFrame]) -> None:
    source_files = []
    for seed in SEEDS:
        for model in MODELS:
            path = SOURCE_ROOT / f"seed_{seed}" / model / "absolute.csv"
            source_files.append(
                {
                    "path": path.relative_to(SOURCE_ROOT).as_posix(),
                    "sha256": sha256(path),
                }
            )
    output_files = []
    for path in sorted(BUILD_ROOT.rglob("*")):
        if path.is_file() and path.name != "package_manifest.json":
            output_files.append(
                {
                    "path": path.relative_to(BUILD_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "package": "stage01_main_kodcode_lora_ablation",
        "score_unit": "percentage_points",
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "arms": list(EXPECTED_ARMS),
        "datasets": list(DATASETS),
        "rows_per_seed_csv": len(MODELS) * len(EXPECTED_ARMS),
        "rows_per_average_csv": len(MODELS) * len(EXPECTED_ARMS),
        "source_files": source_files,
        "output_files": output_files,
        "validation": {
            "source_seed_rows": {
                seed: int(len(frame)) for seed, frame in seed_frames.items()
            },
            "source_overall_recomputed_from_11_tasks": True,
            "base_evaluated_per_seed": True,
        },
    }
    (BUILD_ROOT / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing seed_<seed>/<model>/absolute.csv",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def main() -> None:
    global SOURCE_ROOT, BUILD_ROOT, CSV_DIR, PICTURE_DIR
    args = parse_args()
    SOURCE_ROOT = args.source_root.resolve()
    BUILD_ROOT = args.output_root.resolve()
    CSV_DIR = BUILD_ROOT / "csv"
    PICTURE_DIR = BUILD_ROOT / "picture" / "sft_lora"

    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {SOURCE_ROOT}")
    if BUILD_ROOT in {Path(BUILD_ROOT.anchor), ROOT, SOURCE_ROOT}:
        raise ValueError(f"Unsafe output directory: {BUILD_ROOT}")
    if BUILD_ROOT.exists() and any(BUILD_ROOT.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output is not empty: {BUILD_ROOT}; pass --overwrite to replace it"
            )
        shutil.rmtree(BUILD_ROOT)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)

    seed_frames = {seed: load_seed(seed) for seed in SEEDS}
    for seed, frame in seed_frames.items():
        save_csv(frame, CSV_DIR / f"sft_lora_seed{seed}.csv")

    all_seeds = pd.concat(seed_frames.values(), ignore_index=True)
    key_columns = ["method", "model", "family", "arm"]
    average = (
        all_seeds.groupby(key_columns, sort=False, as_index=False)[
            ["overall_mean_no_health", *DATASETS]
        ]
        .mean()
    )
    average.insert(1, "train_seed", "average")
    average = average[list(CSV_COLUMNS)]
    save_csv(average, CSV_DIR / "sft_lora_average.csv")

    for model, alias in MODELS.items():
        plot_model(average, model, alias)
    write_readme(average)
    validate(seed_frames, average)
    write_package_manifest(seed_frames)

    print(f"Built package: {BUILD_ROOT}")
    print(f"Models: {len(MODELS)}; rows/CSV: {len(average)}; figures: {len(MODELS) * 2}")


if __name__ == "__main__":
    main()
