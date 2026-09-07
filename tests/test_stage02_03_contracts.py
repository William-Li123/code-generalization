from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE02 = REPO_ROOT / "stages/stage02_kodcode_sft_diagnosis"
STAGE03 = REPO_ROOT / "stages/stage03_kodcode_dapo_diagnosis"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_prompt_contract_disables_thinking() -> None:
    module = load_module("stage02_prompt_contract", STAGE02 / "prompt_contract.py")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages, kwargs

    messages, kwargs = module.apply_native_template(
        Tokenizer(), [{"role": "user", "content": "x"}], tokenize=False
    )
    assert messages[0]["content"] == "x"
    assert kwargs["enable_thinking"] is False
    assert kwargs["tokenize"] is False


def test_stage02_paper_row_selector_is_exact_and_fail_closed() -> None:
    module = load_module("stage02_paper_rows", STAGE02 / "select_paper_rows.py")
    rows = [
        {"model": model, "prompt_mode": mode, "checkpoint": checkpoint}
        for model, mode in module.PAPER_PROMPT_MODE.items()
        for checkpoint in module.PAPER_CHECKPOINTS
    ]
    assert len(module.select_paper_rows(rows)) == 8
    with pytest.raises(ValueError, match="missing frozen"):
        module.select_paper_rows(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        module.select_paper_rows(rows + [rows[0]])


def test_generated_hash_contract_detects_mutation(tmp_path: Path) -> None:
    module = load_module("stage03_frozen_contract", STAGE03 / "shared/frozen_contract.py")
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    train_path.write_bytes(b"train")
    val_path.write_bytes(b"val")
    train = [{"extra_info": {"problem_id": "a"}}]
    validation = [{"extra_info": {"problem_id": "b"}}]
    contract = module.generated_contract(train_path, val_path, train, validation)
    module.verify_generated_contract(contract, train_path, val_path, train, validation)
    train_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="train_file_sha256"):
        module.verify_generated_contract(contract, train_path, val_path, train, validation)


def test_dapo_archived_config_records_runtime_dropout_and_counts() -> None:
    config = json.loads(
        (REPO_ROOT / "configs/stage03_kodcode_dapo_diagnosis.json").read_text(encoding="utf-8")
    )
    assert config["shared_training"]["lora_dropout"] == 0.0
    assert "PEFT" in config["shared_training"]["lora_dropout_provenance"]
    assert config["frozen_contract"]["rl10k"]["train_rows"] == 9501
    assert config["frozen_contract"]["full_corpus"]["train_rows"] == 35974


def test_adapter_selection_has_no_implicit_latest_fallback(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    output_root = tmp_path / "output"
    checkpoint_root.mkdir()
    output_root.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(STAGE03 / "shared/select_best_adapter.py"),
            "--checkpoint-root",
            str(checkpoint_root),
            "--output-root",
            str(output_root),
            "--model-key",
            "qwen25",
            "--selection-file",
            str(tmp_path / "selection.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "fail-closed" in result.stderr


def test_appendix_tables_cover_frozen_conditions() -> None:
    module = load_module("appendix_b", STAGE03 / "make_appendix_b_outputs.py")
    data = {}
    for model in module.MODELS:
        data[model] = {}
        for index, condition in enumerate(module.CONDITIONS):
            if condition == "dapo_full" and model != "qwen25":
                continue
            score = 0.5 + index * 0.01
            data[model][condition] = {
                "overall": score,
                "tasks": {task: score for task in module.TASKS},
            }
    table3 = module.table3_rows(data)
    table4 = module.table4_rows(data)
    assert len(table3) == 4
    assert len(table4) == 17
    assert table3[0]["dapo_full"] == 54.0
    assert table3[1]["dapo_full"] is None


def test_rl10k_loop_and_ready_gate_are_present() -> None:
    prepare = (STAGE03 / "rl10k/prepare_rl10k_data.sh").read_text(encoding="utf-8")
    train = (STAGE03 / "rl10k/run_all_train_then_eval.sh").read_text(encoding="utf-8")
    assert "--allow-failures" in prepare
    assert "exact_reward_rejects_cumulative.json" in prepare
    assert ".data_ready" in prepare and ".data_ready" in train


def test_stage03_uses_single_runtime_patch_entrypoint() -> None:
    runner = (STAGE03 / "shared/run_job.sh").read_text(encoding="utf-8")
    readme = (STAGE03 / "README.md").read_text(encoding="utf-8")
    assert "dapo/apply_runtime_patches.py" not in runner
    assert 'bash "$REPO_ROOT/dapo/run_training.sh"' in runner
    assert "VERL_PATCH_DIR=" not in runner
    assert "VLLM_LORA_PATH" not in runner
    assert "CG_VERL_RECIPE_DIR" in readme
    assert "CG_VERL_ENV" in readme
