from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "tests" / "validate_repository.py"
    spec = importlib.util.spec_from_file_location("validate_repository", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_repository_contract() -> None:
    validator = load_validator()
    validator.check_stage_documents()
    # Pytest creates its own caches during collection; the standalone release
    # validator remains strict and rejects those directories before packaging.
    validator.check_files(allow_runtime_caches=True)
    validator.check_json()
    validator.check_release_contracts()


def test_no_empty_executable_files() -> None:
    for suffix in ("*.py", "*.sh"):
        for path in ROOT.rglob(suffix):
            assert path.stat().st_size > 0, path
