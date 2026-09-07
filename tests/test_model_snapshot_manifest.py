from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs" / "model_snapshot_manifest.py"
SPEC = importlib.util.spec_from_file_location("model_snapshot_manifest", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_snapshot_hashes_all_runtime_files(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weight")
    (model / ".cache").mkdir()
    (model / ".cache" / "metadata").write_text("ignored", encoding="utf-8")
    result = MODULE.snapshot(model)
    assert [entry["path"] for entry in result["files"]] == [
        "config.json",
        "model.safetensors",
    ]
    assert result["weight_file_count"] == 1
    assert result["weight_bytes"] == 6


def test_verify_detects_content_change(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    paths = {"model": model}
    reference = MODULE.build(paths)
    MODULE.verify(paths, reference)
    (model / "config.json").write_text('{"changed": true}', encoding="utf-8")
    try:
        MODULE.verify(paths, reference)
    except ValueError as exc:
        assert "differs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("content change was not detected")
