#!/usr/bin/env python3
"""Portable Stage-01 driver for validation, job matrices, and post-processing.

The formal cluster used four independent one-GPU arm lanes.  This driver keeps
that statistical/training contract without embedding a scheduler: it emits 216
single-GPU training jobs and 234 single-GPU evaluation jobs that any local or
cluster array runner can execute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "stage01_main_kodcode_lora_ablation.json"
REFERENCE_MANIFEST = ROOT / "metadata" / "stage01" / "arm_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["validate-inputs", "preflight", "build-matrix", "run-job", "postprocess"]
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-paths",
        type=Path,
        required=True,
        help="JSON object mapping every Stage-01 model key to an external local model directory.",
    )
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--eval-data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--paper-output-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--skip-artifact-sha256", action="store_true")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--label-shuffle-draws", type=int, default=20000)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_external(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the Git checkout: {resolved}")
    return resolved


def load_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    config = read_json(args.config.resolve())
    raw_paths = read_json(args.model_paths.resolve())
    if not isinstance(raw_paths, dict):
        raise ValueError("--model-paths must contain a JSON object")
    expected_keys = set(config["models"])
    if set(raw_paths) != expected_keys:
        raise ValueError(
            f"model-path keys differ: missing={sorted(expected_keys - set(raw_paths))}, "
            f"extra={sorted(set(raw_paths) - expected_keys)}"
        )
    model_paths = {key: Path(value).expanduser().resolve() for key, value in raw_paths.items()}
    for key, path in model_paths.items():
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"model {key} is not a local model snapshot: {path}")

    manifest_path = args.arms_root.resolve() / "manifest.json"
    manifest = read_json(manifest_path)
    reference = read_json(REFERENCE_MANIFEST)
    for field in ("source_sha256", "source_rows", "equal_size_rows", "arm_count"):
        if manifest.get(field) != reference.get(field):
            raise ValueError(
                f"arm manifest {field} mismatch: {manifest.get(field)!r} != {reference.get(field)!r}"
            )
    actual_arms = {entry["name"]: entry for entry in manifest.get("arms", [])}
    reference_arms = {entry["name"]: entry for entry in reference["arms"]}
    if set(actual_arms) != set(reference_arms):
        raise ValueError("arm names differ from the frozen reference")
    for name, expected in reference_arms.items():
        actual = actual_arms[name]
        for field in ("rows", "unique_problem_ids", "problem_id_set_sha256", "parquet_sha256"):
            if actual.get(field) != expected.get(field):
                raise ValueError(f"arm {name} {field} differs from the frozen reference")
        artifact = args.arms_root.resolve() / "arms" / f"{name}.parquet"
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if not args.skip_artifact_sha256 and sha256(artifact) != expected["parquet_sha256"]:
            raise ValueError(f"arm {name} Parquet SHA256 mismatch")

    eval_manifest = read_json(args.eval_data_root.resolve() / "manifest.json")
    if not (
        eval_manifest.get("contract") == "paper_11_task_suite"
        and eval_manifest.get("strict_archived_reference") is True
        and eval_manifest.get("all_tasks_match_archived_reference") is True
    ):
        raise ValueError("evaluation manifest is not the strict frozen paper_11_task_suite")
    return config, model_paths, manifest


def train_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    model_paths: dict[str, Path],
    model_key: str,
    seed: int,
    arm: str,
) -> list[str]:
    training = config["training"]
    model = config["models"][model_key]
    folder = model["folder"]
    return [
        args.python,
        str(STAGE / "train_lora.py"),
        "--model-key", model_key,
        "--model-path", str(model_paths[model_key]),
        "--train-file", str(args.arms_root.resolve() / "arms" / f"{arm}.parquet"),
        "--output-dir", str(args.checkpoint_root.resolve() / f"seed_{seed}" / folder / arm),
        "--prompt-mode", model["prompt_mode"],
        "--target-profile", "auto",
        "--seed", str(seed),
        "--max-seq-len", str(training["max_sequence_length"]),
        "--per-device-batch-size", str(training["per_device_batch_size"]),
        "--gradient-accumulation-steps", str(training["gradient_accumulation_steps"]),
        "--epochs", str(training["epochs"]),
        "--learning-rate", str(training["learning_rate"]),
        "--warmup-ratio", str(training["warmup_ratio"]),
        "--weight-decay", str(training["weight_decay"]),
        "--lora-r", str(training["lora_r"]),
        "--lora-alpha", str(training["lora_alpha"]),
        "--lora-dropout", str(training["lora_dropout"]),
        "--logging-steps", "10",
        "--gradient-checkpointing",
        "--save-resume",
        "--resume",
    ]


def eval_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    model_paths: dict[str, Path],
    model_key: str,
    seed: int,
    arm: str,
) -> list[str]:
    evaluation = config["evaluation"]
    model = config["models"][model_key]
    folder = model["folder"]
    command = [
        args.python,
        str(ROOT / "evaluation" / "paper_suite.py"),
        "--data-root", str(args.eval_data_root.resolve()),
        "--model-name", folder,
        "--base-model-path", str(model_paths[model_key]),
        "--output-dir", str(args.eval_root.resolve() / f"seed_{seed}" / folder / arm),
        "--seed", str(seed),
        "--prompt-mode", evaluation["prompt_mode"],
        "--temperature", str(evaluation["temperature"]),
        "--top-p", str(evaluation["top_p"]),
        "--max-new-tokens-code", str(evaluation["max_new_tokens_code"]),
        "--tasks", ",".join(evaluation["tasks"]),
        "--resume",
    ]
    if arm != "base":
        command.extend(
            [
                "--adapter-path",
                str(
                    args.checkpoint_root.resolve()
                    / f"seed_{seed}"
                    / folder
                    / arm
                    / "final_adapter"
                ),
            ]
        )
    return command


def build_matrix(
    args: argparse.Namespace,
    config: dict[str, Any],
    model_paths: dict[str, Path],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    arms = [entry["name"] for entry in manifest["arms"]]
    jobs: list[dict[str, Any]] = []
    train_ids: dict[tuple[str, int, str], str] = {}
    for model_key in config["models"]:
        for seed in config["training_seeds"]:
            for arm in arms:
                job_id = f"train:{model_key}:{seed}:{arm}"
                train_ids[(model_key, seed, arm)] = job_id
                jobs.append(
                    {
                        "job_id": job_id,
                        "phase": "train",
                        "model_key": model_key,
                        "seed": seed,
                        "arm": arm,
                        "depends_on": [],
                        "command": train_command(args, config, model_paths, model_key, seed, arm),
                    }
                )
    for model_key in config["models"]:
        for seed in config["training_seeds"]:
            for arm in ["base", *arms]:
                jobs.append(
                    {
                        "job_id": f"eval:{model_key}:{seed}:{arm}",
                        "phase": "eval",
                        "model_key": model_key,
                        "seed": seed,
                        "arm": arm,
                        "depends_on": [] if arm == "base" else [train_ids[(model_key, seed, arm)]],
                        "command": eval_command(args, config, model_paths, model_key, seed, arm),
                    }
                )
    for index, job in enumerate(jobs):
        job["index"] = index
        job["shell_preview"] = shlex.join(job["command"])
    matrix = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "formal_contract": {
            "models": len(config["models"]),
            "seeds": len(config["training_seeds"]),
            "training_arms": len(arms),
            "evaluated_conditions": len(arms) + 1,
            "training_jobs": len(config["models"]) * len(config["training_seeds"]) * len(arms),
            "evaluation_jobs": len(config["models"]) * len(config["training_seeds"]) * (len(arms) + 1),
            "gpu_contract": "one visible GPU per matrix job; the historical four-GPU job ran four independent arm jobs concurrently",
        },
        "jobs": jobs,
    }
    return matrix


def run(command: list[str]) -> None:
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def preflight(args: argparse.Namespace, config: dict[str, Any], model_paths: dict[str, Path]) -> None:
    output = args.preflight_output or (args.paper_output_root / "template_preflight.json")
    command = [
        args.python,
        str(STAGE / "preflight_templates.py"),
        "--source", str(args.arms_root.resolve() / "arms" / "full_sft.parquet"),
        "--output", str(assert_external(output, "preflight output")),
        "--max-sequence-length", str(config["training"]["max_sequence_length"]),
    ]
    for key, path in model_paths.items():
        command.extend(["--model", f"{key}={path}"])
    run(command)


def postprocess(args: argparse.Namespace, config: dict[str, Any]) -> None:
    summary_root = assert_external(args.summary_root, "summary root")
    for model in config["models"].values():
        folder = model["folder"]
        for seed in config["training_seeds"]:
            run(
                [
                    args.python,
                    str(ROOT / "analysis" / "summarize_main_results.py"),
                    "--eval-root", str(args.eval_root.resolve() / f"seed_{seed}" / folder),
                    "--output-dir", str(summary_root / f"seed_{seed}" / folder),
                ]
            )
    release = assert_external(args.release_root, "release root")
    run(
        [
            args.python,
            str(ROOT / "analysis" / "package_main_results.py"),
            "--source-root", str(summary_root),
            "--output-root", str(release),
            "--overwrite",
        ]
    )
    run(
        [
            args.python,
            str(ROOT / "analysis" / "reproduce_paper.py"),
            "--package-root", str(release),
            "--output", str(assert_external(args.paper_output_root, "paper output root")),
            "--draws", str(args.draws),
            "--label-shuffle-draws", str(args.label_shuffle_draws),
        ]
    )


def main() -> None:
    args = parse_args()
    for value, label in (
        (args.checkpoint_root, "checkpoint root"),
        (args.eval_root, "evaluation root"),
        (args.summary_root, "summary root"),
        (args.release_root, "release root"),
        (args.paper_output_root, "paper output root"),
        (args.matrix, "matrix path"),
    ):
        assert_external(value, label)
    config, model_paths, manifest = load_contract(args)
    if args.action == "validate-inputs":
        print(json.dumps({"status": "ok", "models": len(model_paths), "arms": len(manifest["arms"])}))
        return
    if args.action == "preflight":
        preflight(args, config, model_paths)
        return
    if args.action == "postprocess":
        postprocess(args, config)
        return
    if args.action == "build-matrix":
        matrix = build_matrix(args, config, model_paths, manifest)
        args.matrix.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.matrix.resolve().write_text(
            json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(matrix["formal_contract"], indent=2))
        return
    if args.job_index is None:
        raise ValueError("run-job requires --job-index")
    matrix = read_json(args.matrix.resolve())
    jobs = matrix.get("jobs", [])
    if args.job_index < 0 or args.job_index >= len(jobs):
        raise IndexError(f"job index {args.job_index} is outside 0..{len(jobs) - 1}")
    run([str(item) for item in jobs[args.job_index]["command"]])


if __name__ == "__main__":
    main()
