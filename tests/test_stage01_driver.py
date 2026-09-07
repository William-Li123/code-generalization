from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def load_driver():
    path = ROOT / "stages" / "stage01_main_kodcode_lora_ablation" / "run_stage01.py"
    spec = importlib.util.spec_from_file_location("stage01_driver", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formal_job_matrix_counts_and_contracts(tmp_path: Path) -> None:
    module = load_driver()
    config = json.loads(
        (ROOT / "configs" / "stage01_main_kodcode_lora_ablation.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (ROOT / "metadata" / "stage01" / "arm_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    args = SimpleNamespace(
        python="python",
        arms_root=tmp_path / "arms",
        checkpoint_root=tmp_path / "checkpoints",
        eval_data_root=tmp_path / "eval_data",
        eval_root=tmp_path / "evaluation",
    )
    model_paths = {key: tmp_path / "models" / key for key in config["models"]}
    matrix = module.build_matrix(args, config, model_paths, manifest)
    assert matrix["formal_contract"]["training_jobs"] == 216
    assert matrix["formal_contract"]["evaluation_jobs"] == 234
    assert len(matrix["jobs"]) == 450
    train = next(job for job in matrix["jobs"] if job["phase"] == "train")
    evaluation = next(
        job for job in matrix["jobs"] if job["phase"] == "eval" and job["arm"] != "base"
    )
    assert train["command"][train["command"].index("--gradient-accumulation-steps") + 1] == "16"
    assert evaluation["command"][evaluation["command"].index("--prompt-mode") + 1] == "chat"
    assert len(evaluation["depends_on"]) == 1
