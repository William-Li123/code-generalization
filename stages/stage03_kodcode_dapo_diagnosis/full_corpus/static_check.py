#!/usr/bin/env python3
"""No-GPU static checks for full-corpus Qwen2.5 DAPO."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


STAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STAGE_ROOT.parents[1]
sys.path.insert(0, str(STAGE_ROOT / "shared"))

from frozen_contract import load_contract, require_equal, verify_generated_contract


def env_path(name: str, *parts: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().joinpath(*parts) if value else None


def parse_args() -> argparse.Namespace:
    default_data_dir = env_path(
        "CG_OUTPUT_ROOT", "stage03_kodcode_dapo_diagnosis", "full_corpus", "data"
    )
    default_model_root = env_path("CG_MODEL_ROOT")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir, required=default_data_dir is None
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=default_model_root,
        required=default_model_root is None,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/stage03_kodcode_dapo_diagnosis.json",
    )
    parser.add_argument(
        "--plain-template",
        type=Path,
        default=STAGE_ROOT / "plain_chat_template.jinja",
    )
    parser.add_argument(
        "--reward-path", type=Path, default=STAGE_ROOT / "shared/kodcode_reward.py"
    )
    parser.add_argument(
        "--run-script", type=Path, default=STAGE_ROOT / "full_corpus/run_train_then_eval.sh"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-train-id-sha256")
    parser.add_argument("--expected-validation-id-sha256")
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="Warn instead of failing for optional frozen ID-hash overrides.",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.data_dir.parent / "audit/static_check.json"
    return args


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    required = [
        args.config,
        args.plain_template,
        args.data_dir / "train.parquet",
        args.data_dir / "val.parquet",
        args.data_dir / "manifest.json",
        args.reward_path,
        args.run_script,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    frozen, branch = load_contract(args.config, "full_corpus")
    require_equal("source_sha256", manifest["source_sha256"], frozen["source_parquet_sha256"])
    require_equal("input_clean_rows", manifest["input_clean_rows"], branch["input_rows"])
    require_equal("input_source_distribution", manifest["input_source_distribution"], branch["source_distribution"])
    final_preflight = json.loads(
        (args.data_dir / "audit/exact_reward_final.json").read_text(encoding="utf-8")
    )
    assert final_preflight["failed"] == 0
    train = pq.read_table(args.data_dir / "train.parquet").to_pylist()
    validation = pq.read_table(args.data_dir / "val.parquet").to_pylist()
    require_equal("train_rows", len(train), branch["train_rows"])
    require_equal("validation_rows", len(validation), branch["validation_rows"])
    verify_generated_contract(
        manifest["generated_contract"],
        args.data_dir / "train.parquet",
        args.data_dir / "val.parquet",
        train,
        validation,
    )
    require_equal(
        "train_id_sha256",
        manifest["generated_contract"]["train_id_sha256"],
        args.expected_train_id_sha256 or branch.get("train_id_sha256"),
        exploratory=args.exploratory,
    )
    require_equal(
        "validation_id_sha256",
        manifest["generated_contract"]["validation_id_sha256"],
        args.expected_validation_id_sha256 or branch.get("validation_id_sha256"),
        exploratory=args.exploratory,
    )
    assert len(train) == manifest["train"]["rows"]
    assert final_preflight["rows"] == len(train) + len(validation)
    assert not (
        {row["extra_info"]["problem_id"] for row in train}
        & {row["extra_info"]["problem_id"] for row in validation}
    )
    assert not (
        {row["extra_info"]["original_question_id"] for row in train}
        & {row["extra_info"]["original_question_id"] for row in validation}
    )
    assert len({row["extra_info"]["original_question_id"] for row in train}) == len(train)
    for row in train[:50] + validation[:50]:
        visible = row["prompt"][0]["content"].lower()
        assert "test_code" not in visible and "canonical_solution" not in visible
        truth = json.loads(row["reward_model"]["ground_truth"])
        assert truth["test_count"] == len(truth["test_functions"]) > 0

    # Verify that training and evaluation both use Qwen2.5's native template.
    sample = train[0]["prompt"]
    checks = {
        "qwen25": (args.model_root / "Qwen2.5-7B-Instruct", "native", "chat"),
    }
    rendered = {}
    prompt_alignment = {}
    plain_template = args.plain_template.read_text(encoding="utf-8")
    for key, (path, train_mode, eval_mode) in checks.items():
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        kwargs = {"chat_template": plain_template} if train_mode == "plain" else {}
        if key.startswith("qwen") and train_mode == "native":
            kwargs["enable_thinking"] = False
        train_text = tokenizer.apply_chat_template(
            sample, tokenize=False, add_generation_prompt=True, **kwargs
        )
        if eval_mode == "chat":
            eval_text = tokenizer.apply_chat_template(
                sample,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            eval_text = sample[0]["content"]
        assert train_text == eval_text, (
            f"{key} training/evaluation prompt mismatch: "
            f"train={train_mode}, eval={eval_mode}"
        )
        rendered[key] = train_text[:200]
        prompt_alignment[key] = {
            "training_mode": train_mode,
            "evaluation_mode": eval_mode,
            "exact_render_match": True,
        }
        if train_mode == "plain":
            assert train_text.strip() == sample[0]["content"].strip()
            assert "<|im_start|>" not in train_text and "<start_of_turn>" not in train_text
        else:
            assert train_text.strip() != sample[0]["content"].strip()

    sys.path.insert(0, str(args.reward_path.parent))
    reward = load_module(args.reward_path)
    truth = {
        "test_code": (
            "from solution import add\n"
            "def test_one():\n    assert add(1, 2) == 3\n"
            "def test_two():\n    assert add(-1, 1) == 0\n"
        ),
        "test_functions": ["test_one", "test_two"],
    }
    good = reward.compute_score("selftest", "def add(a, b):\n    return a + b\n", truth)
    bad = reward.compute_score("selftest", "def add(a, b):\n    return 0\n", truth)
    assert good["score"] == 1.0 and good["passed"] == 1.0
    assert bad["score"] == 0.0 and bad["passed"] == 0.0
    assert bad["pass_frac"] == 0.5
    report = {
        "status": "passed",
        "gpu_training_started": False,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "generated_contract": manifest["generated_contract"],
        "exact_reward_preflight": final_preflight,
        "reward_selftest": {"good": good, "bad": bad},
        "prompt_alignment": prompt_alignment,
        "rendered_prefixes": rendered,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
