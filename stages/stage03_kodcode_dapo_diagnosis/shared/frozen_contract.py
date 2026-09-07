#!/usr/bin/env python3
"""Frozen count/hash contract utilities shared by both Stage-03 data branches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(values: Iterable[object]) -> str:
    """Hash an ordered identifier sequence with an unambiguous JSON encoding."""

    payload = json.dumps([str(value) for value in values], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_contract(config_path: Path, branch: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen = config["frozen_contract"]
    return frozen, frozen[branch]


def require_equal(label: str, actual: Any, expected: Any, *, exploratory: bool = False) -> None:
    if actual == expected or expected is None:
        return
    message = f"frozen contract mismatch for {label}: expected={expected!r}, actual={actual!r}"
    if exploratory:
        print(f"[exploratory-warning] {message}")
    else:
        raise ValueError(message)


def generated_contract(train_path: Path, validation_path: Path, train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> dict[str, Any]:
    def ids(rows: list[dict[str, Any]]) -> list[str]:
        return [str(row["extra_info"]["problem_id"]) for row in rows]

    return {
        "train_file_sha256": sha256_file(train_path),
        "validation_file_sha256": sha256_file(validation_path),
        "train_id_sha256": sha256_ids(ids(train)),
        "validation_id_sha256": sha256_ids(ids(validation)),
    }


def verify_generated_contract(
    contract: dict[str, Any],
    train_path: Path,
    validation_path: Path,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> None:
    actual = generated_contract(train_path, validation_path, train, validation)
    for key, value in actual.items():
        require_equal(key, value, contract.get(key))
