from __future__ import annotations

import hashlib
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "dapo" / "apply_runtime_patches.py"
SPEC = importlib.util.spec_from_file_location("apply_runtime_patches", MODULE_PATH)
assert SPEC and SPEC.loader
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


def test_frozen_runtime_patch_hashes() -> None:
    assert PATCHER.VERL_COMMIT == "e0f4dc2ccdd51e8e445a683b828090577c625534"
    assert len(PATCHER.PATCHES) == 5
    for source, (_, expected) in PATCHER.PATCHES.items():
        data = (ROOT / "dapo" / "runtime_patches" / source).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected


def test_runtime_targets_are_unique_and_scoped_to_verl() -> None:
    targets = [target for target, _ in PATCHER.PATCHES.values()]
    assert len(targets) == len(set(targets))
    assert all(target.startswith("verl/") and target.endswith(".py") for target in targets)


class PatchContractTests(unittest.TestCase):
    def test_patch_hashes(self):
        test_frozen_runtime_patch_hashes()

    def test_scoped_targets(self):
        test_runtime_targets_are_unique_and_scoped_to_verl()
