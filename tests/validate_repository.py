#!/usr/bin/env python3
"""Fast, dependency-free structural validation for the public code tree."""

from __future__ import annotations

import json
import re
import sys
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGES = (
    "stage01_main_kodcode_lora_ablation",
    "stage02_kodcode_sft_diagnosis",
    "stage03_kodcode_dapo_diagnosis",
    "stage04_legacy_taco_lora_ablation",
    "stage05_legacy_taco_dapo_ablation",
    "stage06_legacy_full_sft_robustness",
)
FORBIDDEN_DIRECTORY_NAMES = {
    "__pycache__",
    "checkpoint",
    "checkpoints",
    "data_train",
    "data_test",
    "model",
    "models",
    "output",
    "outputs",
    "logs",
    ".pytest_cache",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".csv",
    ".jsonl",
    ".log",
    ".npy",
    ".npz",
    ".parquet",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
    ".tar",
    ".tgz",
    ".zip",
}
PRIVATE_RUNTIME_PATTERN = re.compile(
    r"(?:/data/" r"yuzheng|/mnt/" r"aoss|[A-Za-z]:[\\/](?:scientific research|SCIENT))",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {".py", ".sh", ".json", ".toml", ".jinja", ".example"}
REFERENCE_RESULT_PREFIX = ("reference_results",)
REQUIRED_RELEASE_FILES = (
    "analysis/make_paper_figures.py",
    "analysis/make_overall_table.py",
    "analysis/make_support_table.py",
    "analysis/make_headline_figure.py",
    "analysis/make_extra_figures.py",
    "analysis/make_residual_null.py",
    "analysis/reproduce_paper.py",
    "analysis/make_calibration_table.py",
    "configs/model_snapshot_manifest.py",
    "metadata/stage01/corpus_manifest.json",
    "metadata/stage01/arm_manifest.json",
    "metadata/evaluation/paper_suite_reference.json",
    "docs/PAPER_TRACEABILITY.md",
    "docs/SOURCE_AUDIT.md",
    "pixi.lock",
    "reference_results/stage01/package_manifest.json",
    "reference_results/appendix_b/package_manifest.json",
    "reference_results/legacy/package_manifest.json",
    "reference_results/stage06/package_manifest.json",
    "stages/stage01_main_kodcode_lora_ablation/run_stage01.py",
)


def fail(message: str) -> None:
    raise AssertionError(message)


def check_stage_documents() -> None:
    expected_docs = {f"{name}.md" for name in EXPECTED_STAGES}
    for directory in (ROOT / "plan", ROOT / "progress"):
        actual = {path.name for path in directory.glob("*.md")}
        if actual != expected_docs:
            fail(f"{directory.name} mismatch: expected {sorted(expected_docs)}, got {sorted(actual)}")
    stage_dirs = {path.name for path in (ROOT / "stages").iterdir() if path.is_dir()}
    if stage_dirs != set(EXPECTED_STAGES):
        fail(f"stage directory mismatch: {sorted(stage_dirs)}")


def check_files(*, allow_runtime_caches: bool = False) -> None:
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        # A public checkout may already be a Git worktree. Repository metadata
        # is neither release content nor an experiment artifact, and may contain
        # paths from the machine that created the checkout.
        if ".git" in relative.parts:
            continue
        if allow_runtime_caches and any(
            part in {"__pycache__", ".pytest_cache"} for part in relative.parts
        ):
            continue
        if path.is_dir():
            if path.name in FORBIDDEN_DIRECTORY_NAMES:
                fail(f"forbidden generated/artifact directory: {relative}")
            continue
        allowed_reference_csv = (
            path.suffix.lower() == ".csv"
            and relative.parts[: len(REFERENCE_RESULT_PREFIX)] == REFERENCE_RESULT_PREFIX
        )
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and not allowed_reference_csv:
            fail(f"forbidden artifact file type: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            fail(f"file exceeds 5 MiB public-code limit: {relative}")
        if path.name.endswith(("~", ".bak", ".orig", ".rej")):
            fail(f"backup/editor file: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".env.example"}:
            text = path.read_text(encoding="utf-8", errors="strict")
            if PRIVATE_RUNTIME_PATTERN.search(text):
                fail(f"private absolute runtime path in {relative}")


def check_json() -> None:
    for path in ROOT.rglob("*.json"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - message path
            fail(f"invalid JSON in {path.relative_to(ROOT)}: {exc}")


def check_python() -> None:
    for path in ROOT.rglob("*.py"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_release_contracts() -> None:
    missing = [relative for relative in REQUIRED_RELEASE_FILES if not (ROOT / relative).is_file()]
    if missing:
        fail(f"required public reproduction files are missing: {missing}")

    arm_manifest = json.loads(
        (ROOT / "metadata" / "stage01" / "arm_manifest.json").read_text(encoding="utf-8")
    )
    if arm_manifest.get("source_rows") != 35974 or arm_manifest.get("arm_count") != 12:
        fail("Stage-01 frozen arm manifest has unexpected counts")
    if len(arm_manifest.get("arms", [])) != 12:
        fail("Stage-01 frozen arm manifest must contain twelve arms")

    result_root = ROOT / "reference_results" / "stage01"
    package = json.loads((result_root / "package_manifest.json").read_text(encoding="utf-8"))
    if package.get("models") != 6 or package.get("evaluated_conditions") != 13:
        fail("reference result package has unexpected model/condition counts")
    expected_columns = {
        "method",
        "train_seed",
        "model",
        "family",
        "arm",
        "overall_mean_no_health",
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
    }
    for entry in package.get("files", []):
        path = result_root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            fail(f"reference result file missing or size mismatch: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            fail(f"reference result SHA256 mismatch: {entry['path']}")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != entry["rows"] or set(rows[0]) != expected_columns:
            fail(f"reference result schema/count mismatch: {entry['path']}")
        pairs = {(row["model"], row["arm"]) for row in rows}
        if len(pairs) != 78:
            fail(f"reference result has duplicate or missing model/arm pairs: {entry['path']}")

    for package_name, expected_files in (
        ("appendix_b", 27),
        ("legacy", 4),
        ("stage06", 3),
    ):
        package_root = ROOT / "reference_results" / package_name
        package_manifest = json.loads(
            (package_root / "package_manifest.json").read_text(encoding="utf-8")
        )
        entries = package_manifest.get("files", [])
        if len(entries) != expected_files:
            fail(
                f"{package_name} reference package expected {expected_files} files, "
                f"got {len(entries)}"
            )
        for entry in entries:
            path = package_root / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                fail(f"{package_name} reference file missing/size mismatch: {entry['path']}")
            if sha256(path) != entry["sha256"]:
                fail(f"{package_name} reference SHA256 mismatch: {entry['path']}")
            if path.suffix == ".csv" and "rows" in entry:
                with path.open(encoding="utf-8", newline="") as handle:
                    if sum(1 for _ in csv.DictReader(handle)) != entry["rows"]:
                        fail(f"{package_name} reference row-count mismatch: {entry['path']}")


def main() -> None:
    check_stage_documents()
    check_files()
    check_json()
    check_python()
    check_release_contracts()
    release_files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    ]
    file_count = len(release_files)
    byte_count = sum(path.stat().st_size for path in release_files)
    print(
        json.dumps(
            {
                "status": "ok",
                "stages": len(EXPECTED_STAGES),
                "files": file_count,
                "bytes": byte_count,
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"repository validation failed: {exc}", file=sys.stderr)
        raise
