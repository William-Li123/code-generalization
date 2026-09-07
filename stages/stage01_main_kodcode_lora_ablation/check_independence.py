#!/usr/bin/env python3
"""Check that Stage 01 code has no private path or bundled-asset dependency."""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_PATHS = [
    STAGE_ROOT / "build_v2.py",
    STAGE_ROOT / "build_arms.py",
    STAGE_ROOT / "train_lora.py",
    STAGE_ROOT / "preflight_templates.py",
    STAGE_ROOT / "check_independence.py",
    STAGE_ROOT / "taxonomy.json",
    REPO_ROOT / "configs" / "stage01_main_kodcode_lora_ablation.json",
    REPO_ROOT / "evaluation" / "prepare_standard11.py",
    REPO_ROOT / "evaluation" / "paper_suite.py",
    REPO_ROOT / "analysis" / "summarize_main_results.py",
]
ABSOLUTE_POSIX = re.compile(r"(?<![A-Za-z0-9_])/(?:data|mnt|home|root|workspace)/")
WINDOWS_DRIVE = re.compile(r"(?i)(?<![A-Za-z0-9_])[a-z]:[\\/]")
LEGACY_PROJECT = "code" + "generalization"
FORBIDDEN_ASSET_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".jsonl",
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
}


def main() -> None:
    errors: list[str] = []
    for path in RUNTIME_PATHS:
        if not path.is_file():
            errors.append(f"missing runtime source: {path.relative_to(REPO_ROOT)}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if ABSOLUTE_POSIX.search(text) or WINDOWS_DRIVE.search(text):
            errors.append(f"hard-coded private absolute path: {path.relative_to(REPO_ROOT)}")
        if LEGACY_PROJECT in text.lower():
            errors.append(f"legacy project dependency: {path.relative_to(REPO_ROOT)}")

    for item in STAGE_ROOT.rglob("*"):
        if item.is_symlink():
            errors.append(f"symlink is not allowed in clean Stage 01: {item.relative_to(REPO_ROOT)}")
        if item.is_file() and item.suffix.lower() in FORBIDDEN_ASSET_SUFFIXES:
            errors.append(f"bundled data/model asset: {item.relative_to(REPO_ROOT)}")

    result = {
        "schema_version": 1,
        "stage": "stage01_main_kodcode_lora_ablation",
        "checked_files": [str(path.relative_to(REPO_ROOT)) for path in RUNTIME_PATHS],
        "passed": not errors,
        "errors": errors,
        "external_inputs_expected": [
            "reviewed taxonomy labels",
            "materialized training rows",
            "model weights or Hugging Face cache entries",
            "paper-suite evaluation datasets",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
