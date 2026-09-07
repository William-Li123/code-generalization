from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "stages" / "stage06_legacy_full_sft_robustness"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figure7_uses_exact_ten_task_contract() -> None:
    package = load("stage06_package", STAGE / "package_appendix_a.py")
    evaluator = load("stage06_eval", STAGE / "evaluate_full_sft_matrix.py")
    assert len(package.TASKS) == 10
    assert set(package.TASKS) == set(evaluator.TASK_N)
    assert "gsm8k" not in package.TASKS
    assert evaluator.CONTRACT == "appendix_a_10_no_gsm8k"


def test_released_stage06_package_is_complete() -> None:
    package = load("stage06_package_reference", STAGE / "package_appendix_a.py")
    root = ROOT / "reference_results" / "stage06"
    per_seed, averaged = package.collect_reference(root)
    assert set(per_seed) == {20260603, 20260604}
    assert all(len(rows) == 20 for rows in per_seed.values())
    assert len(averaged) == 20
    assert [row["arm"] for row in averaged[:10]] == ["base", *package.ARMS]
    qwen25_full = next(
        row
        for row in averaged
        if row["model"] == "Qwen2.5-7B-Instruct" and row["arm"] == "full_sft"
    )
    assert abs(float(qwen25_full["overall_mean_no_health"]) - 25.9554633407) < 1e-8
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_component"] == "03962"
    assert manifest["evaluation_contract"] == "appendix_a_10_no_gsm8k"
