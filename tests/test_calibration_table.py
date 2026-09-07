from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "analysis" / "make_calibration_table.py"
SPEC = importlib.util.spec_from_file_location("make_calibration_table", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_calibration_loaders(tmp_path: Path) -> None:
    main_path = tmp_path / "main.csv"
    with main_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "arm", *MODULE.TASKS.values()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "model": MODULE.MODEL,
                "arm": "base",
                **{column: index for index, column in enumerate(MODULE.TASKS.values())},
            }
        )
    legacy_path = tmp_path / "legacy.csv"
    with legacy_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "arm", "dataset", "score"])
        writer.writeheader()
        for index, display in enumerate(MODULE.TASKS):
            writer.writerow(
                {"model": MODULE.MODEL, "arm": "base", "dataset": display, "score": index + 1}
            )
        writer.writerow({"model": MODULE.MODEL, "arm": "base", "dataset": "Overall", "score": 0})
    assert len(MODULE.load_main(main_path)) == 11
    assert len(MODULE.load_legacy(legacy_path)) == 11
