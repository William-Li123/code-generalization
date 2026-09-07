#!/usr/bin/env python3
"""Validate the Stage 06 matrix and optionally load every full checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any


MODELS = {"Qwen2.5-7B-Instruct", "Qwen3-8B-Base"}
ARMS = {
    "full_sft",
    "full_sft_14043",
    "no_math_number_theory",
    "no_data_structure",
    "no_dynamic_programming",
    "no_greedy_search",
    "no_implementation_simulation",
    "no_graph",
    "no_other_algorithm",
}
SEEDS = {20260603, 20260604}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=sorted(SEEDS))
    parser.add_argument("--load", action="store_true", help="Actually instantiate each checkpoint with from_pretrained.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_matrix(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if (
        payload.get("schema_version") != 1
        or payload.get("job_count") != 36
        or not isinstance(jobs, list)
        or len(jobs) != 36
    ):
        raise ValueError("formal Stage 06 matrix must contain exactly 36 jobs")
    cells = {(int(job["training_seed"]), job["model"], job["arm"]) for job in jobs}
    expected = {(seed, model, arm) for seed in SEEDS for model in MODELS for arm in ARMS}
    if cells != expected:
        missing = sorted(expected - cells)
        extra = sorted(cells - expected)
        raise ValueError(f"Stage 06 matrix mismatch; missing={missing}, extra={extra}")
    return jobs


def inspect_files(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(job["checkpoint_dir"]))
    required = ["config.json", "run_manifest.json", "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).is_file()]
    weights = sorted(path.glob("*.safetensors"))
    if missing or not weights:
        raise FileNotFoundError(f"incomplete checkpoint {path}: missing={missing}, weights={len(weights)}")
    run_manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "seed": int(job["training_seed"]),
        "model_name": job["model"],
        "arm": job["arm"],
    }
    mismatches = {key: (run_manifest.get(key), value) for key, value in expected.items() if run_manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"checkpoint run_manifest mismatch at {path}: {mismatches}")
    return {
        "job_id": job["job_id"],
        "checkpoint_dir": str(path),
        "safetensor_files": len(weights),
        "bytes": sum(item.stat().st_size for item in weights),
        "loaded": False,
    }


def load_checkpoint(path: Path) -> tuple[str, int]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count <= 0:
        raise ValueError(f"loaded model has no parameters: {path}")
    architecture = type(model).__name__
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return architecture, parameter_count


def main() -> None:
    args = parse_args()
    jobs = load_matrix(args.matrix)
    if args.training_seed is not None:
        jobs = [job for job in jobs if int(job["training_seed"]) == args.training_seed]
    rows = []
    for job in jobs:
        row = inspect_files(job)
        if args.load:
            architecture, parameter_count = load_checkpoint(Path(str(job["checkpoint_dir"])))
            row.update({"loaded": True, "architecture": architecture, "parameter_count": parameter_count})
        rows.append(row)
        print(json.dumps({"job_id": row["job_id"], "loaded": row["loaded"]}))
    report = {
        "schema_version": 1,
        "matrix": str(args.matrix.resolve()),
        "mode": "actual_from_pretrained_load" if args.load else "files_and_run_manifest_only",
        "validated": len(rows),
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validated": len(rows), "mode": report["mode"], "output": str(args.output) if args.output else None}))


if __name__ == "__main__":
    main()
