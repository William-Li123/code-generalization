#!/usr/bin/env python3
"""Build deterministic, equal-size Stage 1 SFT ablation arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic Stage 1 arms from an external labeled Parquet file."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Prefer a repository-relative path without rejecting external datasets."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def derived_seed(seed: int, name: str) -> int:
    raw = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(raw[:4], "big")


def category_counts(frame: pd.DataFrame) -> dict[str, int]:
    values = frame["primary_category"].value_counts().sort_index()
    return {str(key): int(value) for key, value in values.items()}


def frame_identity(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for value in frame["problem_id"].astype(str).sort_values():
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def choose(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(frame) < count:
        raise ValueError(f"candidate has {len(frame)} rows, expected at least {count}")
    if len(frame) == count:
        return frame.copy()
    return frame.sample(n=count, replace=False, random_state=seed)


def write_arm(
    frame: pd.DataFrame,
    name: str,
    arms_dir: Path,
    ids_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output = arms_dir / f"{name}.parquet"
    frame = frame.sort_values("problem_id", kind="stable").reset_index(drop=True)
    frame.to_parquet(output, index=False)
    ids_path = ids_dir / f"{name}.txt"
    ids_path.write_text("\n".join(frame["problem_id"].astype(str)) + "\n", encoding="utf-8")
    return {
        "name": name,
        "rows": int(len(frame)),
        "unique_problem_ids": int(frame["problem_id"].nunique()),
        "category_counts": category_counts(frame),
        "problem_id_set_sha256": frame_identity(frame),
        "parquet": portable_path(output),
        "parquet_sha256": sha256(output),
        "ids": portable_path(ids_path),
        **metadata,
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"--source must be an existing labeled Parquet file outside the code checkout: {source}"
        )
    arms_dir = output_dir / "arms"
    ids_dir = output_dir / "arm_ids"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists; pass --overwrite to rebuild")
    arms_dir.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(source)
    required = {"problem_id", "prompt", "response", "primary_category"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"source is missing columns: {missing}")
    if frame["problem_id"].duplicated().any():
        duplicates = frame.loc[frame["problem_id"].duplicated(), "problem_id"].head().tolist()
        raise ValueError(f"problem_id must be unique; examples={duplicates}")
    if frame[list(required)].isna().any().any():
        raise ValueError("required columns contain null values")

    source_counts = category_counts(frame)
    categories = sorted(source_counts)
    equal_size = min(len(frame) - count for count in source_counts.values())
    largest = max(source_counts, key=source_counts.get)
    if equal_size != len(frame) - source_counts[largest]:
        raise AssertionError("equal-size derivation is inconsistent")

    arms: list[dict[str, Any]] = []
    arms.append(
        write_arm(
            frame,
            "full_sft",
            arms_dir,
            ids_dir,
            {"role": "full_data_reference", "removed_category": None, "sampling_seed": None},
        )
    )
    control_name = f"full_sft_{equal_size}"
    control = choose(frame, equal_size, derived_seed(args.seed, control_name))
    if set(control["primary_category"]) != set(categories):
        raise ValueError("equal-size random control does not cover every category")
    arms.append(
        write_arm(
            control,
            control_name,
            arms_dir,
            ids_dir,
            {
                "role": "equal_size_random_control",
                "removed_category": None,
                "sampling_seed": derived_seed(args.seed, control_name),
            },
        )
    )

    for category in categories:
        name = f"no_{category}"
        candidates = frame.loc[frame["primary_category"] != category]
        sampled = choose(candidates, equal_size, derived_seed(args.seed, name))
        if category in set(sampled["primary_category"]):
            raise AssertionError(f"{name} still contains removed category")
        arms.append(
            write_arm(
                sampled,
                name,
                arms_dir,
                ids_dir,
                {
                    "role": "leave_one_category_out",
                    "removed_category": category,
                    "candidate_rows_after_removal": int(len(candidates)),
                    "sampling_seed": derived_seed(args.seed, name),
                },
            )
        )

    manifest = {
        "schema_version": 1,
        "experiment": "stage1_kodcode_v2_sft_code_ablation",
        "source": portable_path(source),
        "source_sha256": sha256(source),
        "source_rows": int(len(frame)),
        "source_unique_problem_ids": int(frame["problem_id"].nunique()),
        "source_category_counts": source_counts,
        "data_seed": args.seed,
        "equal_size_rows": equal_size,
        "largest_category": largest,
        "arm_count": len(arms),
        "arms": arms,
        "primary_comparison": f"{control_name} versus each no_<category> arm",
        "full_data_reference": "full_sft",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
