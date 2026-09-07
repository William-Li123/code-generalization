#!/usr/bin/env python3
"""Rebuild Appendix-B Tables 3/4 and Figures 8/9 from frozen raw metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen25", "qwen3", "llama31", "gemma2")
MODEL_LABELS = {
    "qwen25": "Qwen2.5-7B-Instruct",
    "qwen3": "Qwen3-8B-Base",
    "llama31": "Llama-3.1-8B-Instruct",
    "gemma2": "Gemma-2-9B-Instruct",
}
PAPER_MODE = {"qwen25": "native", "qwen3": "plain", "llama31": "native", "gemma2": "native"}
EVAL_MODE = {"qwen25": "chat", "qwen3": "plain", "llama31": "chat", "gemma2": "chat"}
TRANSFER_TASKS = ("arc_challenge", "finqa", "legalbench", "medcalc", "scienceqa")
IN_DOMAIN_TASKS = ("humaneval", "mbpp_plus", "mbpp_simple", "math500_medium", "math500_high_level", "gsm8k")
TASKS = TRANSFER_TASKS + IN_DOMAIN_TASKS
CONDITIONS = ("base", "sft_half", "sft_final", "dapo_rl10k", "dapo_full")
CONDITION_LABELS = {
    "base": "Base",
    "sft_half": "SFT 0.5 epoch",
    "sft_final": "SFT 1.0 epoch",
    "dapo_rl10k": "DAPO 9.6K",
    "dapo_full": "DAPO 36K",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(payload: dict[str, Any], task: str) -> float:
    found: list[float] = []
    for section in ("main", "hard", "diagnostic"):
        value = payload.get(section, {}).get(task)
        if value is not None:
            raw = value.get("accuracy", value.get("pass_at_1"))
            if raw is None:
                raise ValueError(f"task {task} has no accuracy/pass_at_1")
            found.append(float(raw))
    if len(found) != 1:
        raise ValueError(f"task {task} must map to exactly one metric section; found={len(found)}")
    return found[0]


def load_metrics(path: Path, expected_prompt_mode: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("prompt_mode") != expected_prompt_mode:
        raise ValueError(
            f"prompt mode mismatch in {path}: expected={expected_prompt_mode}, actual={payload.get('prompt_mode')}"
        )
    if payload.get("scoreable_dataset_count") != 11:
        raise ValueError(f"expected 11 scoreable datasets in {path}")
    task_values = {task: metric(payload, task) for task in TASKS}
    mean = sum(task_values.values()) / len(task_values)
    if abs(mean - float(payload["scoreable_overall_mean"])) > 1e-10:
        raise ValueError(f"overall mean does not equal the 11-task mean in {path}")
    return {
        "path": path,
        "sha256": sha256(path),
        "overall": mean,
        "tasks": task_values,
    }


def verify_selection(path: Path, model: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_key") != model:
        raise ValueError(f"selection model mismatch in {path}")
    if payload.get("reason") != "best_validation_full_pass_rate":
        raise ValueError(f"formal selection is not validation-selected in {path}")
    if payload.get("n") != 100:
        raise ValueError(f"selection must use exactly 100 validation rows in {path}")
    if payload.get("step") is None or not payload.get("validation_file"):
        raise ValueError(f"incomplete selection record in {path}")
    return payload


def load_inputs(stage02_work: Path, rl10k_work: Path, full_work: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    data: dict[str, dict[str, dict[str, Any]]] = {}
    inputs: dict[str, Any] = {"metrics": {}, "selection": {}}
    for model in MODELS:
        mode = PAPER_MODE[model]
        expected = EVAL_MODE[model]
        paths = {
            "base": stage02_work / f"results/formal/{model}/{mode}/base/metrics.json",
            "sft_half": stage02_work / f"results/formal/{model}/{mode}/half_epoch/metrics.json",
            "sft_final": stage02_work / f"results/formal/{model}/{mode}/final/metrics.json",
            "dapo_rl10k": rl10k_work / f"results/formal/{model}/dapo_best/metrics.json",
        }
        if model == "qwen25":
            paths["dapo_full"] = full_work / "results/formal/qwen25/dapo_best/metrics.json"
        data[model] = {condition: load_metrics(path, expected) for condition, path in paths.items()}
        inputs["metrics"][model] = {
            condition: {"path": str(value["path"]), "sha256": value["sha256"]}
            for condition, value in data[model].items()
        }

        rl_selection_path = rl10k_work / f"results/selection/{model}.json"
        inputs["selection"][f"rl10k/{model}"] = verify_selection(rl_selection_path, model)
        rl_base = load_metrics(rl10k_work / f"results/formal/{model}/base/metrics.json", expected)
        data[model]["dapo_rl10k_base"] = rl_base
        inputs["metrics"][model]["dapo_rl10k_base"] = {
            "path": str(rl_base["path"]),
            "sha256": rl_base["sha256"],
        }
        inputs.setdefault("baseline_replicate_drift_pp", {})[f"rl10k/{model}"] = {
            task: round(
                100.0
                * abs(
                    (data[model]["base"][task] if task == "overall" else data[model]["base"]["tasks"][task])
                    - (rl_base[task] if task == "overall" else rl_base["tasks"][task])
                ),
                9,
            )
            for task in ("overall", *TASKS)
        }

    full_selection_path = full_work / "results/selection/qwen25.json"
    inputs["selection"]["full_corpus/qwen25"] = verify_selection(full_selection_path, "qwen25")
    full_base = load_metrics(full_work / "results/formal/qwen25/base/metrics.json", "chat")
    data["qwen25"]["dapo_full_base"] = full_base
    inputs["metrics"]["qwen25"]["dapo_full_base"] = {
        "path": str(full_base["path"]),
        "sha256": full_base["sha256"],
    }
    inputs.setdefault("baseline_replicate_drift_pp", {})["full_corpus/qwen25"] = {
        task: round(
            100.0
            * abs(
                (data["qwen25"]["base"][task] if task == "overall" else data["qwen25"]["base"]["tasks"][task])
                - (full_base[task] if task == "overall" else full_base["tasks"][task])
            ),
            9,
        )
        for task in ("overall", *TASKS)
    }
    return data, inputs


def baseline_for(
    data: dict[str, dict[str, dict[str, Any]]], model: str, condition: str
) -> dict[str, Any]:
    """Return the manuscript's shared Stage-02 Base reference.

    The archived SFT, RL10K-DAPO, and full-corpus-DAPO branches each evaluated
    the same base snapshot again at nonzero decoding temperature.  Their Base
    scores therefore differ by sampling noise.  Tables 3/4 and Figure 9 use the
    single frozen Stage-02 Base level printed in the manuscript, rather than
    silently switching the denominator between conditions.  The independent
    DAPO Base replicates remain hash-addressed in the output manifest and their
    drift is reported there for auditability.
    """

    del condition
    return data[model]["base"]


def percent(value: float) -> float:
    return 100.0 * value


def group_mean(metrics: dict[str, Any], tasks: tuple[str, ...]) -> float:
    return sum(metrics["tasks"][task] for task in tasks) / len(tasks)


def table3_rows(data: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        row: dict[str, Any] = {"model_key": model, "model": MODEL_LABELS[model]}
        for condition in CONDITIONS:
            row[condition] = (
                round(percent(data[model][condition]["overall"]), 6)
                if condition in data[model]
                else None
            )
        rows.append(row)
    return rows


def table4_rows(data: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            if condition not in data[model]:
                continue
            paired_base = baseline_for(data, model, condition)
            base_transfer = group_mean(paired_base, TRANSFER_TASKS)
            base_domain = group_mean(paired_base, IN_DOMAIN_TASKS)
            transfer = group_mean(data[model][condition], TRANSFER_TASKS)
            domain = group_mean(data[model][condition], IN_DOMAIN_TASKS)
            rows.append(
                {
                    "model_key": model,
                    "model": MODEL_LABELS[model],
                    "condition": condition,
                    "transfer_percent": round(percent(transfer), 6),
                    "in_domain_percent": round(percent(domain), 6),
                    "transfer_delta_vs_base": round(percent(transfer - base_transfer), 6),
                    "in_domain_delta_vs_base": round(percent(domain - base_domain), 6),
                }
            )
    return rows


def check_reference(table3: list[dict[str, Any]], table4: list[dict[str, Any]], reference_path: Path) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    ref3 = reference["table3_overall_percent"]
    for row in table3:
        for condition in CONDITIONS:
            expected = ref3[row["model_key"]].get(condition)
            actual = row[condition]
            if expected is None and actual is None:
                continue
            if expected is None or actual is None or round(float(actual), 1) != float(expected):
                raise ValueError(
                    f"Table 3 reference mismatch for {row['model_key']}/{condition}: expected={expected}, actual={actual}"
                )
    by_key = {(row["model_key"], row["condition"]): row for row in table4}
    ref4 = reference["table4_group_percent"]
    mapping = {
        "sft_half": "sft_half_delta",
        "sft_final": "sft_final_delta",
        "dapo_rl10k": "dapo_rl10k_delta",
        "dapo_full": "dapo_full_delta",
    }
    for model, expected in ref4.items():
        base = by_key[(model, "base")]
        if [round(base["transfer_percent"], 1), round(base["in_domain_percent"], 1)] != expected["base"]:
            raise ValueError(f"Table 4 base reference mismatch for {model}")
        for condition, ref_key in mapping.items():
            if ref_key not in expected:
                continue
            row = by_key[(model, condition)]
            actual = [round(row["transfer_delta_vs_base"], 1), round(row["in_domain_delta_vs_base"], 1)]
            if actual != expected[ref_key]:
                raise ValueError(f"Table 4 delta reference mismatch for {model}/{condition}: {actual}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_figure(data: dict[str, dict[str, dict[str, Any]]], path: Path, *, delta: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    columns = (*TASKS, "overall")
    fig, axes = plt.subplots(2, 2, figsize=(18, 8), constrained_layout=True)
    image = None
    for axis, model in zip(axes.flat, MODELS, strict=True):
        conditions = [condition for condition in CONDITIONS if condition in data[model]]
        values = []
        for condition in conditions:
            row = [data[model][condition]["tasks"][task] for task in TASKS]
            row.append(data[model][condition]["overall"])
            if delta:
                paired_base = baseline_for(data, model, condition)
                base = [paired_base["tasks"][task] for task in TASKS]
                base.append(paired_base["overall"])
                row = [value - baseline for value, baseline in zip(row, base, strict=True)]
            values.append([percent(value) for value in row])
        matrix = np.asarray(values)
        vmax = max(1.0, float(np.abs(matrix).max())) if delta else 100.0
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r" if delta else "YlGnBu", vmin=-vmax if delta else 0, vmax=vmax)
        axis.set_title(MODEL_LABELS[model])
        axis.set_yticks(range(len(conditions)), [CONDITION_LABELS[value] for value in conditions])
        axis.set_xticks(range(len(columns)), columns, rotation=55, ha="right", fontsize=8)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                axis.text(x, y, f"{matrix[y, x]:+.1f}" if delta else f"{matrix[y, x]:.1f}", ha="center", va="center", fontsize=6)
    assert image is not None
    fig.colorbar(image, ax=axes, label="Delta vs Base (percentage points)" if delta else "Score (%)", shrink=0.75)
    fig.suptitle("Appendix B: delta vs Base" if delta else "Appendix B: absolute 11-task results")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage02-work", type=Path, required=True)
    parser.add_argument("--rl10k-work", type=Path, required=True)
    parser.add_argument("--full-work", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO_ROOT / "metadata/stage03/appendix_b_reference.json",
    )
    parser.add_argument(
        "--exploratory-allow-paper-value-drift",
        action="store_true",
        help="Keep strict path/prompt/selection checks but skip rounded paper-value checks.",
    )
    args = parser.parse_args()
    data, inputs = load_inputs(args.stage02_work, args.rl10k_work, args.full_work)
    table3 = table3_rows(data)
    table4 = table4_rows(data)
    if not args.exploratory_allow_paper_value_drift:
        check_reference(table3, table4, args.reference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "table3.csv", table3)
    write_csv(args.output_dir / "table4.csv", table4)
    render_figure(data, args.output_dir / "figure8_absolute.png", delta=False)
    render_figure(data, args.output_dir / "figure9_delta.png", delta=True)
    manifest = {
        "contract": "appendix_b_tables3_4_figures8_9_v1",
        "strict_paper_reference_checked": not args.exploratory_allow_paper_value_drift,
        "inputs": inputs,
        "outputs": {
            name: {"path": str(args.output_dir / name), "sha256": sha256(args.output_dir / name)}
            for name in ("table3.csv", "table4.csv", "figure8_absolute.png", "figure9_delta.png")
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
