#!/usr/bin/env python3
"""Prepare the exact 11-task paper evaluation layout without private paths.

Four datasets can be normalized from public upstream sources. The historical
project copied the remaining normalized artifacts from an older checkout whose
normalization code and pinned revisions were not preserved. To avoid silently
changing the benchmark, those files must be supplied through ``--source-root``.
Datasets are written outside this code repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable


GSM8K_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_SOURCE_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_REVISION}/grade_school_math/data/test.jsonl"
)
GSM8K_SOURCE_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"

REFERENCE: dict[str, dict[str, Any]] = {
    "arc_challenge": {
        "path": "main_test/arc_challenge.jsonl",
        "rows": 1172,
        "sha256": "3f3b45d20ad67c78bd4936aa73c80312048b5cd97d3e825d57ca21228357c30e",
    },
    "finqa": {
        "path": "main_test/finqa_official_test.jsonl",
        "rows": 1147,
        "sha256": "6c8c1968fc4eae6f3734834a30675a939cfedbff603c979512471d4c0596bcfc",
    },
    "humaneval": {
        "path": "main_test/humaneval.jsonl",
        "rows": 164,
        "sha256": "c8cb0035b250f1399a6b7068ed4e65addd0a68d19d348ab350964b552bc02e4b",
    },
    "legalbench": {
        "path": "main_test/legalbench_rule_application_test.jsonl",
        "rows": 1689,
        "sha256": "15680415b1521f99298a75d33aac9d7b2b1b0277ada19d0b4ba770b27a7d500c",
    },
    "math500_medium": {
        "path": "main_test/math500_medium.jsonl",
        "rows": 105,
        "sha256": "726a804a64c9cf63b948159fa73dc01c36f675d2e8e8d464083ec0fa3dc98aa0",
    },
    "mbpp_plus": {
        "path": "main_test/mbpp_plus.jsonl",
        "rows": 378,
        "sha256": "2a38ee610526fc27b45a1b67426f27b8fc9f4b134ac932ac4ed65b09f90166f6",
    },
    "medcalc": {
        "path": "main_test/medcalc_bench_verified_test.jsonl",
        "rows": 1100,
        "sha256": "10141832c4327477603d00b7f24bcb24e9892cb265051ea92162cca90845be7e",
    },
    "math500_high_level": {
        "path": "hard_test/math500_high_level.jsonl",
        "rows": 262,
        "sha256": "111ea34ab2c7ef4c5ef19735be9c1cbe3ea1a4dfcc77af2286c58aefb635a249",
    },
    "gsm8k": {
        "path": "diagnostic_test/gsm8k.jsonl",
        "rows": 1319,
        "sha256": "fb8df8e5d83a0f75e4cc781c46507d2d298c55c8d9c52e003c96583eaa20368f",
    },
    "mbpp_simple": {
        "path": "diagnostic_test/mbpp_simple.jsonl",
        "rows": 257,
        "sha256": "bcc197414b924ffea07cabe55d1cad80ad3a5428fe46600c49afadc839cec066",
    },
    "scienceqa": {
        "path": "diagnostic_test/scienceqa.jsonl",
        "rows": 2224,
        "sha256": "9606844d1fb78a0578216a1ae7d4b20b78907eb1a4cc4c8ceaffda4dd8f2f3be",
    },
}

EXTERNAL_CANDIDATES = {
    "arc_challenge": [
        "main_test/arc_challenge.jsonl",
        "data_test/main_test/arc_challenge.jsonl",
        "test_data/auxiliary/arc_challenge.jsonl",
        "auxiliary/arc_challenge.jsonl",
        "arc_challenge.jsonl",
    ],
    "finqa": [
        "main_test/finqa_official_test.jsonl",
        "data_test/main_test/finqa_official_test.jsonl",
        "test_data/main/finqa_official_test.jsonl",
        "main/finqa_official_test.jsonl",
        "finqa_official_test.jsonl",
    ],
    "humaneval": [
        "main_test/humaneval.jsonl",
        "data_test/main_test/humaneval.jsonl",
        "test_data/auxiliary/humaneval.jsonl",
        "auxiliary/humaneval.jsonl",
        "humaneval.jsonl",
    ],
    "legalbench": [
        "main_test/legalbench_rule_application_test.jsonl",
        "data_test/main_test/legalbench_rule_application_test.jsonl",
        "test_data/main/legalbench_rule_application_test.jsonl",
        "main/legalbench_rule_application_test.jsonl",
        "legalbench_rule_application_test.jsonl",
    ],
    "medcalc": [
        "main_test/medcalc_bench_verified_test.jsonl",
        "data_test/main_test/medcalc_bench_verified_test.jsonl",
        "test_data/main/medcalc_bench_verified_test.jsonl",
        "main/medcalc_bench_verified_test.jsonl",
        "medcalc_bench_verified_test.jsonl",
    ],
}

MATH_MEDIUM_CANDIDATES = [
    "main_test/math500_medium.jsonl",
    "data_test/main_test/math500_medium.jsonl",
    "math500_medium.jsonl",
]
MATH_HIGH_CANDIDATES = [
    "hard_test/math500_high_level.jsonl",
    "data_test/hard_test/math500_high_level.jsonl",
    "math500_high_level.jsonl",
]
MATH_COMBINED_CANDIDATES = [
    "test_data/auxiliary/math500.jsonl",
    "auxiliary/math500.jsonl",
    "math500.jsonl",
]

PUBLIC_NORMALIZED_CANDIDATES = {
    task: [spec["path"], f"data_test/{spec['path']}", Path(spec["path"]).name]
    for task, spec in REFERENCE.items()
    if task in {"gsm8k", "mbpp_plus", "mbpp_simple", "scienceqa"}
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "External tree containing preserved normalized JSONL files. Required "
            "for ARC-C, FinQA, HumanEval, LegalBench, MedCalc, and MATH-500."
        ),
    )
    parser.add_argument("--mbppplus-revision")
    parser.add_argument("--mbpp-revision")
    parser.add_argument("--scienceqa-revision")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download public datasets; require normalized copies under --source-root.",
    )
    parser.add_argument(
        "--allow-nonreference-content",
        action="store_true",
        help="Allow row/hash mismatches for exploratory runs; mismatches remain recorded.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_candidate(root: Path, candidates: list[str]) -> Path | None:
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path.resolve()
    return None


def relative_source(path: Path, source_root: Path) -> str:
    try:
        return path.relative_to(source_root).as_posix()
    except ValueError:
        return path.name


def resolve_external_sources(
    source_root: Path | None,
) -> tuple[Path, dict[str, Path], tuple[str, Path | tuple[Path, Path]]]:
    if source_root is None:
        raise FileNotFoundError(
            "The archived workflow did not preserve exact public normalizers/revisions for "
            "ARC-C, FinQA, HumanEval, LegalBench, MedCalc, and MATH-500. Supply "
            "--source-root containing their preserved normalized JSONL files; see --help "
            "for accepted layouts."
        )
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--source-root is not a directory: {root}")

    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for task, candidates in EXTERNAL_CANDIDATES.items():
        path = find_candidate(root, candidates)
        if path is None:
            missing.append(f"{task}: one of {', '.join(candidates)}")
        else:
            resolved[task] = path

    medium = find_candidate(root, MATH_MEDIUM_CANDIDATES)
    high = find_candidate(root, MATH_HIGH_CANDIDATES)
    if medium is not None and high is not None:
        math_source: tuple[str, Path | tuple[Path, Path]] = ("split", (medium, high))
    else:
        combined = find_candidate(root, MATH_COMBINED_CANDIDATES)
        if combined is None:
            missing.append(
                "math500: provide both normalized medium/high files or one combined "
                + ", ".join(MATH_COMBINED_CANDIDATES)
            )
            math_source = ("missing", root)
        else:
            math_source = ("combined", combined)

    if missing:
        raise FileNotFoundError(
            "--source-root is incomplete:\n- " + "\n- ".join(missing)
        )
    return root, resolved, math_source


def copy_jsonl(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copyfile(source, destination)


def normalize_gsm8k(destination: Path) -> dict[str, Any]:
    with urllib.request.urlopen(GSM8K_SOURCE_URL, timeout=120) as response:
        source_bytes = response.read()
    actual_source_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_source_hash != GSM8K_SOURCE_SHA256:
        raise ValueError(
            f"unexpected GSM8K source hash: expected {GSM8K_SOURCE_SHA256}, "
            f"got {actual_source_hash}"
        )
    source_rows = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for index, row in enumerate(source_rows):
        answer_cot = str(row["answer"])
        if "####" not in answer_cot:
            raise ValueError(f"GSM8K row {index} has no canonical final-answer delimiter")
        rows.append(
            {
                "id": f"gsm8k_{index:05d}",
                "question": row["question"],
                "answer_cot": answer_cot,
                "answer": answer_cot.rsplit("####", 1)[1].strip().replace(",", ""),
                "benchmark": "openai_gsm8k",
                "split": "test",
            }
        )
    write_jsonl(destination, rows)
    return {
        "source": GSM8K_SOURCE_URL,
        "source_revision": GSM8K_REVISION,
        "source_sha256": actual_source_hash,
    }


def load_hf_dataset(
    name: str,
    config: str | None,
    split: str,
    revision: str | None,
) -> Any:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(name, config, **kwargs) if config else load_dataset(name, **kwargs)


def normalize_mbpp_plus(destination: Path, revision: str | None) -> dict[str, Any]:
    rows = []
    for row in load_hf_dataset("evalplus/mbppplus", None, "test", revision):
        rows.append(
            {
                "id": f"mbppplus_{row['task_id']}",
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row.get("code", ""),
                "test_imports": row.get("test_imports") or [],
                "test_list": row.get("test_list") or [],
                "plus_test": row.get("test", ""),
                "benchmark": "mbppplus",
            }
        )
    write_jsonl(destination, rows)
    return {"source": "evalplus/mbppplus", "split": "test", "revision": revision}


def normalize_mbpp_simple(destination: Path, revision: str | None) -> dict[str, Any]:
    rows = []
    dataset = load_hf_dataset(
        "google-research-datasets/mbpp", "sanitized", "test", revision
    )
    for row in dataset:
        rows.append(
            {
                "id": f"mbpp_simple_{row['task_id']}",
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row.get("code", ""),
                "test_imports": row.get("test_imports") or [],
                "test_list": row.get("test_list") or [],
                "benchmark": "mbpp_sanitized",
            }
        )
    write_jsonl(destination, rows)
    return {
        "source": "google-research-datasets/mbpp",
        "config": "sanitized",
        "split": "test",
        "revision": revision,
    }


def normalize_scienceqa(destination: Path, revision: str | None) -> dict[str, Any]:
    rows = []
    for index, row in enumerate(
        load_hf_dataset("derek-thomas/ScienceQA", None, "test", revision)
    ):
        if row.get("image") is not None:
            continue
        choices = list(row["choices"])
        answer_index = int(row["answer"])
        rows.append(
            {
                "id": f"scienceqa_{index:05d}",
                "question": row["question"],
                "choices": choices,
                "answer": answer_index,
                "answer_text": choices[answer_index],
                "hint": row.get("hint") or "",
                "subject": row.get("subject") or "",
                "topic": row.get("topic") or "",
                "category": row.get("category") or "",
                "skill": row.get("skill") or "",
                "benchmark": "scienceqa_text_only",
            }
        )
    write_jsonl(destination, rows)
    return {
        "source": "derek-thomas/ScienceQA",
        "split": "test text-only",
        "revision": revision,
    }


def verify_artifact(task: str, path: Path, allow_nonreference: bool) -> dict[str, Any]:
    expected = REFERENCE[task]
    rows = len(read_jsonl(path))
    digest = sha256(path)
    matches = rows == expected["rows"] and digest == expected["sha256"]
    if not matches and not allow_nonreference:
        raise RuntimeError(
            f"{task} does not match the archived paper artifact: expected "
            f"rows={expected['rows']} sha256={expected['sha256']}, got rows={rows} "
            f"sha256={digest}. Supply the preserved normalized file under --source-root, "
            "or pass --allow-nonreference-content only for an explicitly non-reproduction run."
        )
    return {
        "path": expected["path"],
        "rows": rows,
        "sha256": digest,
        "reference_rows": expected["rows"],
        "reference_sha256": expected["sha256"],
        "matches_archived_reference": matches,
    }


def prepare_public_task(
    task: str,
    destination: Path,
    source_root: Path,
    offline: bool,
    downloader: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    preserved = find_candidate(source_root, PUBLIC_NORMALIZED_CANDIDATES[task])
    if preserved is not None:
        copy_jsonl(preserved, destination)
        return {
            "source": f"source-root:{relative_source(preserved, source_root)}",
            "pre_normalized": True,
        }
    if offline:
        raise FileNotFoundError(
            f"--offline requires a preserved normalized {task} file under --source-root; "
            f"accepted paths: {', '.join(PUBLIC_NORMALIZED_CANDIDATES[task])}"
        )
    return downloader(destination)


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists; pass --overwrite to rebuild")

    source_root, external, math_source = resolve_external_sources(args.source_root)
    task_metadata: dict[str, dict[str, Any]] = {}

    for task, source in external.items():
        destination = output_root / REFERENCE[task]["path"]
        copy_jsonl(source, destination)
        task_metadata[task] = {
            "source": f"source-root:{relative_source(source, source_root)}",
            "pre_normalized": True,
        }

    math_mode, math_payload = math_source
    if math_mode == "split":
        medium_source, high_source = math_payload
        copy_jsonl(medium_source, output_root / REFERENCE["math500_medium"]["path"])
        copy_jsonl(high_source, output_root / REFERENCE["math500_high_level"]["path"])
        task_metadata["math500_medium"] = {
            "source": f"source-root:{relative_source(medium_source, source_root)}",
            "pre_normalized": True,
        }
        task_metadata["math500_high_level"] = {
            "source": f"source-root:{relative_source(high_source, source_root)}",
            "pre_normalized": True,
        }
    elif math_mode == "combined":
        combined_source = math_payload
        rows = read_jsonl(combined_source)
        medium = [row for row in rows if int(row.get("level", 0)) == 3]
        high = [row for row in rows if int(row.get("level", 0)) >= 4]
        write_jsonl(output_root / REFERENCE["math500_medium"]["path"], medium)
        write_jsonl(output_root / REFERENCE["math500_high_level"]["path"], high)
        source_name = f"source-root:{relative_source(combined_source, source_root)}"
        task_metadata["math500_medium"] = {
            "source": source_name,
            "filter": "level == 3",
        }
        task_metadata["math500_high_level"] = {
            "source": source_name,
            "filter": "level >= 4",
        }
    else:
        raise AssertionError("external source resolution returned an invalid MATH-500 mode")

    public_downloaders: dict[str, Callable[[Path], dict[str, Any]]] = {
        "gsm8k": normalize_gsm8k,
        "mbpp_plus": lambda path: normalize_mbpp_plus(path, args.mbppplus_revision),
        "mbpp_simple": lambda path: normalize_mbpp_simple(path, args.mbpp_revision),
        "scienceqa": lambda path: normalize_scienceqa(path, args.scienceqa_revision),
    }
    for task, downloader in public_downloaders.items():
        destination = output_root / REFERENCE[task]["path"]
        task_metadata[task] = prepare_public_task(
            task, destination, source_root, args.offline, downloader
        )

    tasks: dict[str, dict[str, Any]] = {}
    for task in REFERENCE:
        artifact = verify_artifact(
            task,
            output_root / REFERENCE[task]["path"],
            args.allow_nonreference_content,
        )
        tasks[task] = {**artifact, **task_metadata[task]}

    counts: Counter[str] = Counter()
    for entry in tasks.values():
        counts[Path(entry["path"]).parent.name] += int(entry["rows"])
    manifest = {
        "schema_version": 1,
        "contract": "paper_11_task_suite",
        "strict_archived_reference": not args.allow_nonreference_content,
        "all_tasks_match_archived_reference": all(
            entry["matches_archived_reference"] for entry in tasks.values()
        ),
        "tasks": tasks,
        "row_totals": dict(counts),
        "paper_benchmark_groups": {
            "code_generation": ["humaneval", "mbpp_simple", "mbpp_plus"],
            "mathematics": ["gsm8k", "math500_medium", "math500_high_level"],
            "finance": ["finqa"],
            "medical": ["medcalc"],
            "science_qa": ["arc_challenge", "scienceqa"],
            "legal": ["legalbench"],
        },
        "excluded": ["apps_hard", "healthbench", "planbench"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
