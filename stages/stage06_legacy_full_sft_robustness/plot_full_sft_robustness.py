#!/usr/bin/env python3
"""Plot Stage 06 full-SFT robustness scores from portable JSON or CSV inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


AUTO_SCORE_KEYS = (
    "metrics.scoreable_overall_mean_no_health",
    "metrics.scoreable_overall_mean",
    "metrics.main.main_scoreable_mean_no_health",
    "scoreable_overall_mean_no_health",
    "scoreable_overall_mean",
    "main.main_scoreable_mean_no_health",
    "macro_mean",
    "score",
)


@dataclass(frozen=True)
class Observation:
    model: str
    arm: str
    seed: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="JSON or CSV result files.")
    parser.add_argument("--output", type=Path, required=True, help="Destination PNG, SVG, PDF, or JPG.")
    parser.add_argument(
        "--score-key",
        help="Optional dotted key for the score. Known legacy aggregate keys are detected by default.",
    )
    parser.add_argument("--title", default="Legacy full-SFT robustness")
    parser.add_argument("--percent", action="store_true", help="Display scores on a 0-100 scale.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def dotted_get(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def result_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("models", "rows", "runs", "results", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload]


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".json":
        return result_records(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"unsupported input format: {path}")


def first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = dotted_get(record, key)
        if value not in (None, ""):
            return value
    return None


def infer_seed(record: dict[str, Any], source: Path) -> str:
    value = first_present(record, ("seed", "metrics.seed", "training_seed"))
    if value is not None:
        return str(value)
    match = re.search(r"seed[_-]?(\d+)", source.stem, flags=re.IGNORECASE)
    return match.group(1) if match else source.stem


def to_observation(record: dict[str, Any], source: Path, score_key: str | None) -> Observation | None:
    raw_name = first_present(record, ("model", "model_name", "metrics.model_name"))
    raw_arm = first_present(record, ("arm", "ablation_arm", "metrics.arm"))
    if raw_name is None:
        return None
    model = str(raw_name)
    arm = str(raw_arm) if raw_arm is not None else ""
    if not arm and "__" in model:
        model, arm = model.split("__", 1)
    if not arm:
        arm = "full_sft"

    keys = (score_key,) if score_key else AUTO_SCORE_KEYS
    raw_score = first_present(record, keys)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return Observation(model=model, arm=arm, seed=infer_seed(record, source), score=score)


def load_observations(paths: list[Path], score_key: str | None) -> list[Observation]:
    observations: list[Observation] = []
    for path in paths:
        for record in read_records(path):
            observation = to_observation(record, path, score_key)
            if observation is not None:
                observations.append(observation)
    if not observations:
        requested = score_key or ", ".join(AUTO_SCORE_KEYS)
        raise ValueError(f"no plottable scores found; checked: {requested}")
    return observations


def plot(observations: list[Observation], output: Path, title: str, percent: bool, dpi: int) -> None:
    # Matplotlib is intentionally optional for data inspection and job generation.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(item.model for item in observations))
    arms = list(dict.fromkeys(item.arm for item in observations))
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in observations:
        grouped[(item.model, item.arm)].append(item.score * (100.0 if percent else 1.0))

    figure_width = max(9.0, 0.8 * len(arms) + 3.0)
    fig, ax = plt.subplots(figsize=(figure_width, 5.8), constrained_layout=True)
    width = 0.8 / max(1, len(models))
    centers = list(range(len(arms)))
    for model_index, model in enumerate(models):
        offset = (model_index - (len(models) - 1) / 2.0) * width
        means: list[float] = []
        errors: list[float] = []
        for arm in arms:
            values = grouped.get((model, arm), [])
            means.append(statistics.fmean(values) if values else math.nan)
            errors.append(statistics.stdev(values) if len(values) > 1 else 0.0)
        ax.bar(
            [center + offset for center in centers],
            means,
            width=width,
            yerr=errors,
            capsize=3,
            label=model,
        )

    ax.set_title(title)
    ax.set_ylabel("Macro score (%)" if percent else "Macro score")
    ax.set_xticks(centers, arms, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    observations = load_observations(args.input, args.score_key)
    plot(observations, args.output, args.title, args.percent, args.dpi)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "observations": len(observations),
                "models": sorted({item.model for item in observations}),
                "arms": sorted({item.arm for item in observations}),
                "seeds": sorted({item.seed for item in observations}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
