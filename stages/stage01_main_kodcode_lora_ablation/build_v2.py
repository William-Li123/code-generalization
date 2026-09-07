#!/usr/bin/env python3
"""Deterministically map reviewed v1 labels to the public v2 taxonomy.

The historical script discovered ``classification_v1`` through the project
layout. This release keeps data outside the repository, so every data location
is supplied explicitly. ``--labels`` accepts reviewed JSONL, JSON, or Parquet
labels. ``--v1-root`` is a convenience for an intact legacy classification
export and fails with a useful inventory when it is incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_TAXONOMY = HERE / "taxonomy.json"
VERSION = "2.0.0"

DIRECT_MAP = {
    "string_and_parsing": "string_and_parsing",
    "dynamic_programming": "dynamic_programming",
    "graph_and_tree_algorithms": "graph_structures_and_stateful_systems",
    "data_structures_and_queries": "graph_structures_and_stateful_systems",
    "simulation_and_stateful_logic": "graph_structures_and_stateful_systems",
    "greedy_search_and_optimization": "greedy_search_and_optimization",
    "math_and_number_theory": "math_and_number_theory",
    "direct_implementation_and_utilities": "direct_implementation_and_utilities",
}

ARRAY_MAP = {
    "hashing_counting": "hashing_counting_and_sets",
    "sequence_transformation": "sequence_transformations",
    "sorting": "sorting_and_ordered_processing",
    "sliding_window": "range_window_and_matrix_processing",
    "two_pointers": "range_window_and_matrix_processing",
    "prefix_sum": "range_window_and_matrix_processing",
    "matrix": "range_window_and_matrix_processing",
}

MATERIALIZED_ARGUMENTS = {
    "train/sft": "train_sft",
    "train/rl": "train_rl",
    "validation/sft": "validation_sft",
    "validation/rl": "validation_rl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--labels",
        type=Path,
        help="Reviewed v1 labels in .parquet, .jsonl, or .json format.",
    )
    source.add_argument(
        "--v1-root",
        type=Path,
        help=(
            "Root of an intact legacy classification_v1 export. Expected files "
            "are state/labels.parquet and output/{train,validation}/{sft,rl}_labeled.parquet."
        ),
    )
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sft", type=Path)
    parser.add_argument("--train-rl", type=Path)
    parser.add_argument("--validation-sft", type=Path)
    parser.add_argument("--validation-rl", type=Path)
    parser.add_argument(
        "--expected-rows",
        type=int,
        help="Optional provenance assertion; no corpus size is assumed by default.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, role: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {role}: {path}. Supply an existing reviewed JSONL/Parquet "
            "file explicitly, or point --v1-root at a complete classification export."
        )
    return path


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("rows", payload.get("data"))
        if not isinstance(payload, list):
            raise ValueError(f"JSON input must be a list or contain a rows/data list: {path}")
        return pd.DataFrame(payload)
    raise ValueError(f"Unsupported input format for {path}; use .parquet, .jsonl, or .json")


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, dict[str, Path]]:
    supplied = {
        label: getattr(args, attribute)
        for label, attribute in MATERIALIZED_ARGUMENTS.items()
        if getattr(args, attribute) is not None
    }
    if args.v1_root is None:
        labels = require_file(args.labels, "reviewed label table")
        return labels, {
            label: require_file(path, f"materialized {label} table")
            for label, path in supplied.items()
        }

    root = args.v1_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"--v1-root does not exist or is not a directory: {root}. "
            "Use --labels for a standalone reviewed JSONL/Parquet export."
        )
    labels = require_file(root / "state" / "labels.parquet", "legacy reviewed label table")
    defaults = {
        label: root
        / "output"
        / label.split("/", 1)[0]
        / f"{label.split('/', 1)[1]}_labeled.parquet"
        for label in MATERIALIZED_ARGUMENTS
    }
    defaults.update(supplied)
    return labels, {
        label: require_file(path, f"legacy materialized {label} table")
        for label, path in defaults.items()
    }


def map_category(primary: str, subcategory: str) -> str:
    if primary == "array_hashing_and_sorting":
        if subcategory not in ARRAY_MAP:
            raise ValueError(f"unmapped v1 array subcategory: {subcategory}")
        return ARRAY_MAP[subcategory]
    if primary not in DIRECT_MAP:
        raise ValueError(f"unmapped v1 primary category: {primary}")
    return DIRECT_MAP[primary]


def transform(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    aliases = {
        "v1_primary_category": "primary_category",
        "v1_subcategory": "subcategory",
        "v1_confidence": "confidence",
        "v1_alternative_category": "alternative_category",
        "v1_taxonomy_version": "taxonomy_version",
        "v1_first_pass_category": "first_pass_category",
    }
    for destination, source in aliases.items():
        if destination not in result.columns and source in result.columns:
            result[destination] = result[source]
    required = {"problem_id", "v1_primary_category", "v1_subcategory"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(
            "reviewed labels are missing columns "
            f"{missing}; expected problem_id plus primary_category/subcategory "
            "(or their v1_-prefixed equivalents)"
        )
    result["primary_category"] = [
        map_category(str(primary), str(subcategory))
        for primary, subcategory in zip(
            result["v1_primary_category"], result["v1_subcategory"], strict=True
        )
    ]
    result["subcategory"] = result["v1_subcategory"]
    if "v1_confidence" in result.columns:
        result["confidence"] = result["v1_confidence"]
    result["taxonomy_version"] = VERSION
    result["mapping_method"] = "deterministic_v1_primary_subcategory_map"
    return result


def category_metrics(counts: pd.Series) -> dict[str, float]:
    if counts.empty or (counts <= 0).any():
        raise ValueError("category counts must be non-empty and positive")
    proportions = counts / counts.sum()
    entropy = 1.0
    if len(counts) > 1:
        entropy = float(
            -(proportions * proportions.map(math.log)).sum() / math.log(len(counts))
        )
    return {
        "category_count": int(len(counts)),
        "largest_count": int(counts.max()),
        "smallest_count": int(counts.min()),
        "largest_share": float(proportions.max()),
        "smallest_share": float(proportions.min()),
        "max_min_ratio": float(counts.max() / counts.min()),
        "normalized_entropy": entropy,
    }


def write_distribution(labels: pd.DataFrame, output_root: Path) -> dict[str, Any]:
    audit = output_root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    v1_counts = labels["v1_primary_category"].value_counts().sort_index()
    v2_counts = labels["primary_category"].value_counts().sort_values(ascending=False)
    distribution = v2_counts.rename_axis("primary_category").reset_index(name="count")
    distribution["percent"] = distribution["count"] / len(labels) * 100
    distribution.to_csv(audit / "category_distribution.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(
        labels["v1_primary_category"], labels["primary_category"], margins=True
    ).to_csv(audit / "v1_to_v2_transition.csv", encoding="utf-8-sig")

    figure, axis = plt.subplots(figsize=(13, 6.5))
    plot = distribution.sort_values("count")
    bars = axis.barh(plot["primary_category"], plot["count"], color="#3274a1")
    axis.bar_label(
        bars,
        labels=[
            f"{count:,} ({percent:.1f}%)"
            for count, percent in zip(plot["count"], plot["percent"], strict=True)
        ],
        padding=5,
        fontsize=9,
    )
    axis.set_title("Code capability distribution: balanced taxonomy v2")
    axis.set_xlabel("Problems")
    axis.set_xlim(0, distribution["count"].max() * 1.22)
    figure.tight_layout()
    figure.savefig(audit / "category_distribution.png", dpi=180)
    plt.close(figure)

    metrics = {"v1": category_metrics(v1_counts), "v2": category_metrics(v2_counts)}
    (audit / "balance_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def unique_techniques(labels: pd.DataFrame) -> int:
    if "techniques" not in labels.columns:
        return 0
    tags: set[str] = set()
    for value in labels["techniques"]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if hasattr(value, "tolist"):
            value = value.tolist()
        values = value if isinstance(value, (list, tuple, set)) else [value]
        tags.update(str(tag) for tag in values if str(tag))
    return len(tags)


def main() -> None:
    args = parse_args()
    labels_path, materialized = resolve_inputs(args)
    taxonomy_path = require_file(args.taxonomy, "taxonomy")
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    allowed = {str(item["id"]) for item in taxonomy["categories"]}

    labels = transform(read_table(labels_path))
    if labels["problem_id"].duplicated().any():
        examples = labels.loc[labels["problem_id"].duplicated(), "problem_id"].head().tolist()
        raise RuntimeError(f"v2 labels must have unique problem IDs; examples={examples}")
    if args.expected_rows is not None and len(labels) != args.expected_rows:
        raise RuntimeError(f"expected {args.expected_rows} label rows, found {len(labels)}")
    actual_categories = set(labels["primary_category"].astype(str))
    if actual_categories != allowed:
        raise RuntimeError(
            "v2 output categories do not match taxonomy; "
            f"missing={sorted(allowed - actual_categories)}, extra={sorted(actual_categories - allowed)}"
        )

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    labels_output = output_root / "labels_v2.parquet"
    labels.to_parquet(labels_output, index=False)
    labels.drop(columns=["techniques"], errors="ignore").to_csv(
        output_root / "labels_v2.csv", index=False, encoding="utf-8-sig"
    )

    outputs: dict[str, dict[str, Any]] = {}
    sft_ids: set[str] = set()
    materialized_sft_keys: set[str] = set()
    for key, source in materialized.items():
        split, kind = key.split("/", 1)
        destination = output_root / split / f"{kind}_labeled.parquet"
        frame = transform(read_table(source))
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
        if kind == "sft":
            sft_ids.update(frame["problem_id"].astype(str))
            materialized_sft_keys.add(key)
        outputs[key] = {
            "source_name": source.name,
            "source_sha256": sha256(source),
            "rows": int(len(frame)),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "path": destination.relative_to(output_root).as_posix(),
        }

    expected_sft_keys = {"train/sft", "validation/sft"}
    sft_coverage_verified = materialized_sft_keys == expected_sft_keys
    if sft_coverage_verified and sft_ids != set(labels["problem_id"].astype(str)):
        raise RuntimeError("materialized train+validation SFT IDs do not match v2 labels")

    metrics = write_distribution(labels, output_root)
    manifest = {
        "schema_version": 1,
        "taxonomy": taxonomy["name"],
        "taxonomy_version": VERSION,
        "taxonomy_sha256": sha256(taxonomy_path),
        "derivation": taxonomy["derivation"],
        "api_calls": 0,
        "source_labels_name": labels_path.name,
        "source_labels_sha256": sha256(labels_path),
        "rows": int(len(labels)),
        "unique_problem_ids": int(labels["problem_id"].nunique()),
        "primary_tags_v1": int(labels["v1_primary_category"].nunique()),
        "subcategory_tags_v1_present": int(labels["v1_subcategory"].nunique()),
        "technique_tags_v1_unique": unique_techniques(labels),
        "primary_tags_v2": int(labels["primary_category"].nunique()),
        "balance": metrics,
        "labels_v2": {
            "path": labels_output.relative_to(output_root).as_posix(),
            "sha256": sha256(labels_output),
        },
        "materialized_outputs": outputs,
        "sft_id_coverage_verified": sft_coverage_verified,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(labels["primary_category"].value_counts().to_string())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
