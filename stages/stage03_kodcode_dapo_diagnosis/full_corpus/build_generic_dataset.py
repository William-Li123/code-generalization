#!/usr/bin/env python3
"""Export the verified KodCode split into template-neutral SFT and RL datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sft", type=Path, required=True)
    parser.add_argument("--source-rl-train", type=Path, required=True)
    parser.add_argument("--source-rl-validation", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def normalize_messages(value: Any) -> list[dict[str, str]]:
    value = as_python(value)
    if not isinstance(value, list):
        raise TypeError(f"messages must be a list, got {type(value)!r}")
    messages = []
    for message in value:
        if hasattr(message, "as_py"):
            message = message.as_py()
        if not isinstance(message, dict):
            message = dict(message)
        messages.append(
            {"role": str(message["role"]), "content": str(message["content"])}
        )
    return messages


def normalize_extra_info(extra: Any, index: int) -> dict[str, Any]:
    if hasattr(extra, "as_py"):
        extra = extra.as_py()
    if not isinstance(extra, dict):
        extra = dict(extra)
    return {
        "index": index,
        "problem_id": str(extra["problem_id"]),
        "original_question_id": str(extra["original_question_id"]),
        "source_dataset": str(extra["source_dataset"]),
        "subset": str(extra["subset"]),
        "style": str(extra["style"]),
        "difficulty": str(extra["difficulty"]),
        "entry_point": str(extra["entry_point"]),
        "test_count_used": int(extra["test_count_used"]),
        "test_modules": list(as_python(extra.get("test_modules", []))),
        "test_code_repaired": bool(extra["test_code_repaired"]),
        "source_full_execution_verified": bool(
            extra["source_full_execution_verified"]
        ),
    }


def build_split(
    rl_path: Path,
    source_by_id: dict[str, dict[str, Any]],
    output_dir: Path,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rl_rows = pq.read_table(rl_path).to_pylist()
    sft_rows: list[dict[str, Any]] = []
    generic_rl_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rl_rows):
        prompt_messages = normalize_messages(row["prompt"])
        if len(prompt_messages) != 1 or prompt_messages[0]["role"] != "user":
            raise ValueError(
                f"{split}[{index}] expected exactly one user prompt message"
            )
        extra = normalize_extra_info(row["extra_info"], index)
        problem_id = extra["problem_id"]
        if problem_id not in source_by_id:
            raise KeyError(f"{split}[{index}] missing SFT source for {problem_id}")
        source = source_by_id[problem_id]
        sft_prompt = str(source["prompt"])
        response = str(source["answer"])
        if not sft_prompt.strip():
            raise ValueError(f"{split}[{index}] has an empty SFT prompt")
        if not response.strip():
            raise ValueError(f"{split}[{index}] has an empty SFT response")

        common_metadata = {
            "problem_id": problem_id,
            "original_question_id": extra["original_question_id"],
            "source_dataset": extra["source_dataset"],
            "subset": extra["subset"],
            "style": extra["style"],
            "difficulty": extra["difficulty"],
            "entry_point": extra["entry_point"],
            "test_count_used": extra["test_count_used"],
            "test_code_repaired": extra["test_code_repaired"],
            "reference_full_execution_verified": True,
        }
        sft_rows.append(
            {
                **common_metadata,
                "messages": [
                    {"role": "user", "content": sft_prompt},
                    {"role": "assistant", "content": response},
                ],
                "prompt": sft_prompt,
                "response": response,
                "response_source": str(source.get("response_source", "")),
                "quality_status": str(source.get("quality_status", "")),
            }
        )

        reward_model = row["reward_model"]
        if hasattr(reward_model, "as_py"):
            reward_model = reward_model.as_py()
        if not isinstance(reward_model, dict):
            reward_model = dict(reward_model)
        generic_rl_rows.append(
            {
                "prompt": prompt_messages,
                "data_source": "kodcode_clean_36074",
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(reward_model["ground_truth"]),
                },
                "extra_info": extra,
            }
        )

    split_output = output_dir / split
    split_output.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(sft_rows),
        split_output / "sft.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(generic_rl_rows),
        split_output / "rl.parquet",
        compression="zstd",
    )
    return sft_rows, generic_rl_rows


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def main() -> None:
    args = parse_args()
    if not args.ready_marker.is_file():
        raise SystemExit(f"verified data marker is missing: {args.ready_marker}")

    source = pd.read_parquet(args.source_sft)
    if source["id"].astype(str).duplicated().any():
        duplicates = source.loc[
            source["id"].astype(str).duplicated(), "id"
        ].head()
        raise ValueError(f"SFT source IDs are not unique: {duplicates.tolist()}")
    source_by_id = {
        str(row["id"]): row for row in source.to_dict(orient="records")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_sft, train_rl = build_split(
        args.source_rl_train, source_by_id, args.output_dir, "train"
    )
    val_sft, val_rl = build_split(
        args.source_rl_validation,
        source_by_id,
        args.output_dir,
        "validation",
    )

    train_ids = {row["problem_id"] for row in train_sft}
    validation_ids = {row["problem_id"] for row in val_sft}
    train_original_ids = {row["original_question_id"] for row in train_sft}
    validation_original_ids = {
        row["original_question_id"] for row in val_sft
    }
    if train_ids & validation_ids:
        raise ValueError("train and validation problem IDs overlap")
    if train_original_ids & validation_original_ids:
        raise ValueError("train and validation original-question IDs overlap")
    if len(train_sft) != len(train_rl) or len(val_sft) != len(val_rl):
        raise ValueError("SFT and RL row counts do not match")
    if len(train_sft) + len(val_sft) != 36074:
        raise ValueError("expected exactly 36,074 selected examples")

    output_files = [
        args.output_dir / "train/sft.parquet",
        args.output_dir / "train/rl.parquet",
        args.output_dir / "validation/sft.parquet",
        args.output_dir / "validation/rl.parquet",
    ]
    manifest = {
        "name": "kodcode_clean_36074",
        "version": "1.0",
        "template_policy": (
            "Template-neutral structured messages. Apply each model's tokenizer "
            "chat template only in the training or inference pipeline."
        ),
        "counts": {
            "total": len(train_sft) + len(val_sft),
            "train": len(train_sft),
            "validation": len(val_sft),
            "sft_total": len(train_sft) + len(val_sft),
            "rl_total": len(train_rl) + len(val_rl),
        },
        "train_distribution": {
            "source_dataset": distribution(train_sft, "source_dataset"),
            "subset": distribution(train_sft, "subset"),
            "style": distribution(train_sft, "style"),
            "difficulty": distribution(train_sft, "difficulty"),
        },
        "validation_distribution": {
            "source_dataset": distribution(val_sft, "source_dataset"),
            "subset": distribution(val_sft, "subset"),
            "style": distribution(val_sft, "style"),
            "difficulty": distribution(val_sft, "difficulty"),
        },
        "verification": {
            "all_reference_answers_passed_all_hidden_tests": True,
            "strict_binary_reward": "1 iff all tests pass; otherwise 0",
            "train_validation_problem_id_overlap": 0,
            "train_validation_original_question_overlap": 0,
            "hidden_tests_present_in_sft_files": False,
            "hidden_tests_present_in_rl_reward_model": True,
        },
        "source_files": {
            "sft_sha256": sha256(args.source_sft),
            "rl_train_sha256": sha256(args.source_rl_train),
            "rl_validation_sha256": sha256(args.source_rl_validation),
        },
        "output_files": {
            str(path.relative_to(args.output_dir)).replace("\\", "/"): {
                "rows": pq.read_metadata(path).num_rows,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_files
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
