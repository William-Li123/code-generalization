#!/usr/bin/env python3
"""Evaluate Stage 1 base models and LoRA adapters on the copied test suites."""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
_DATA_TEST_ENV = os.environ.get("CG_DATA_TEST_ROOT")
DATA_TEST = Path(_DATA_TEST_ENV).expanduser() if _DATA_TEST_ENV else Path()
MAIN = DATA_TEST / "main_test"
HARD = DATA_TEST / "hard_test"
DIAG = DATA_TEST / "diagnostic_test"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?%?")
ANSWER_IS = re.compile(r"(?:final\s+answer|the\s+answer|answer)\s*(?:is|=|:)\s*([^\n]+)", re.IGNORECASE)
HASH_ANSWER = re.compile(r"####\s*([^\n]+)")
BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
LATEX_FRAC = re.compile(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}")
SLASH_FRAC = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?\s*/\s*-?\d+(?:\.\d+)?(?![\w.])")
FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
DEFAULT_APPS_CODE_MAX_CHARS = 200_000
DEFAULT_APPS_CODE_MAX_LINES = 8_000
DEFAULT_APPS_CODE_RECORD_CHARS = 20_000

MAIN_TASKS = [
    "arc_challenge",
    "finqa",
    "healthbench",
    "healthbench_professional",
    "humaneval",
    "legalbench",
    "math500_medium",
    "mbpp_plus",
    "medcalc",
]
HARD_TASKS = ["apps_hard", "math500_high_level"]
DIAGNOSTIC_TASKS = ["gsm8k", "mbpp_simple", "scienceqa"]
# Canonical paper suite. APPS-Hard, HealthBench and PlanBench remain available
# to dedicated evaluators, but they are not part of the paper's aggregate.
PAPER_SCOREABLE_TASKS = [
    "arc_challenge",
    "finqa",
    "humaneval",
    "legalbench",
    "math500_medium",
    "mbpp_plus",
    "medcalc",
    "math500_high_level",
    "gsm8k",
    "mbpp_simple",
    "scienceqa",
]
SCOREABLE_TASKS = PAPER_SCOREABLE_TASKS
UNSCORED_GENERATION_TASKS = ["healthbench", "healthbench_professional"]

GSM8K_EXPECTED_ROWS = 1319
GSM8K_SOURCE_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_SOURCE_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"

GSM_SHOTS = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether. The answer is 72.

Q: Weng earns $12 an hour for babysitting. Yesterday, she did 50 minutes of babysitting. How much did she earn?
A: She earns 12/60 = 0.2 dollars per minute. She earned 0.2 * 50 = 10 dollars. The answer is 10.

"""

APPS_HARNESS = r'''
import io
import json
import math
import sys
import traceback

payload = json.loads(sys.stdin.read())
code = payload["code"]
tests = payload["tests"]
fn_name = payload.get("fn_name")

class FakeStdin:
    def __init__(self, text):
        self._text = io.StringIO(text)
        self.buffer = io.BytesIO(text.encode())
    def read(self, *args):
        return self._text.read(*args)
    def readline(self, *args):
        return self._text.readline(*args)
    def readlines(self, *args):
        return self._text.readlines(*args)
    def __iter__(self):
        return iter(self._text)

def normalize_stdout(text):
    text = "" if text is None else str(text)
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()

def stdout_equal(got, expected):
    a = normalize_stdout(got)
    b = normalize_stdout(expected)
    return a == b or a.split() == b.split()

def unwrap_expected(value):
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value

def value_equal(a, b):
    b = unwrap_expected(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(value_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a.keys()) == set(b.keys()) and all(value_equal(a[k], b[k]) for k in a)
    return str(a) == str(b)

try:
    if fn_name:
        ns = {"__name__": "__main__"}
        exec(compile(code, "<solution>", "exec"), ns)
        fn = ns.get(fn_name)
        if not callable(fn):
            print(json.dumps({"passed": False, "status": "missing_function", "passed_tests": 0, "total_tests": len(tests)}))
            raise SystemExit
        for idx, test in enumerate(tests):
            args = test["input"]
            if not isinstance(args, list):
                args = [args]
            got = fn(*args)
            if not value_equal(got, test["output"]):
                print(json.dumps({"passed": False, "status": "wrong_answer", "failed_test": idx, "passed_tests": idx, "total_tests": len(tests), "got": repr(got)[:300], "expected": repr(test["output"])[:300]}))
                raise SystemExit
        print(json.dumps({"passed": True, "status": "passed", "passed_tests": len(tests), "total_tests": len(tests)}))
    else:
        for idx, test in enumerate(tests):
            old_stdin, old_stdout = sys.stdin, sys.stdout
            fake_out = io.StringIO()
            ns = {"__name__": "__main__"}
            try:
                sys.stdin = FakeStdin(str(test["input"]))
                sys.stdout = fake_out
                try:
                    exec(compile(code, "<solution>", "exec"), ns)
                except SystemExit:
                    pass
            finally:
                sys.stdin, sys.stdout = old_stdin, old_stdout
            got = fake_out.getvalue()
            if not stdout_equal(got, test["output"]):
                print(json.dumps({"passed": False, "status": "wrong_answer", "failed_test": idx, "passed_tests": idx, "total_tests": len(tests), "got": normalize_stdout(got)[:300], "expected": normalize_stdout(test["output"])[:300]}))
                raise SystemExit
        print(json.dumps({"passed": True, "status": "passed", "passed_tests": len(tests), "total_tests": len(tests)}))
except SyntaxError as exc:
    print(json.dumps({"passed": False, "status": "syntax_error", "passed_tests": 0, "total_tests": len(tests), "error": str(exc)[:300]}))
except Exception as exc:
    print(json.dumps({"passed": False, "status": "runtime_error", "passed_tests": 0, "total_tests": len(tests), "error": str(exc)[:300], "traceback": traceback.format_exc()[-500:]}))
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(_DATA_TEST_ENV).expanduser() if _DATA_TEST_ENV else None,
        required=_DATA_TEST_ENV is None,
        help="External data_test directory; may also be set with CG_DATA_TEST_ROOT.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prompt-mode", choices=["plain", "chat", "auto"], default="chat")
    parser.add_argument("--health-prompt-mode", choices=["plain", "chat", "auto"], default="chat")
    parser.add_argument("--choice-batch-size", type=int, default=24)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--apps-batch-size", type=int, default=2)
    parser.add_argument("--health-batch-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--health-temperature", type=float, default=0.0)
    parser.add_argument("--health-top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens-code", type=int, default=2048)
    parser.add_argument("--max-new-tokens-apps", type=int, default=2048)
    parser.add_argument("--max-new-tokens-health", type=int, default=2048)
    parser.add_argument("--apps-exec-timeout", type=float, default=12.0)
    parser.add_argument("--apps-max-tests-per-problem", type=int, default=0)
    parser.add_argument("--apps-code-max-chars", type=int, default=DEFAULT_APPS_CODE_MAX_CHARS)
    parser.add_argument("--apps-code-max-lines", type=int, default=DEFAULT_APPS_CODE_MAX_LINES)
    parser.add_argument("--apps-code-record-chars", type=int, default=DEFAULT_APPS_CODE_RECORD_CHARS)
    parser.add_argument(
        "--tasks",
        default="scoreable",
        help="Task list or alias. Default is the canonical 11-task paper suite; use 'all' only for explicit legacy/supplementary runs.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per task; 0 means full task.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-healthbench-generation", action="store_true")
    parser.add_argument(
        "--allow-unverified-data",
        action="store_true",
        help="Permit an exploratory run without the strict archived-reference manifest.",
    )
    return parser.parse_args()


PAPER_REFERENCE = {
    "arc_challenge": ("main_test/arc_challenge.jsonl", 1172, "3f3b45d20ad67c78bd4936aa73c80312048b5cd97d3e825d57ca21228357c30e"),
    "finqa": ("main_test/finqa_official_test.jsonl", 1147, "6c8c1968fc4eae6f3734834a30675a939cfedbff603c979512471d4c0596bcfc"),
    "humaneval": ("main_test/humaneval.jsonl", 164, "c8cb0035b250f1399a6b7068ed4e65addd0a68d19d348ab350964b552bc02e4b"),
    "legalbench": ("main_test/legalbench_rule_application_test.jsonl", 1689, "15680415b1521f99298a75d33aac9d7b2b1b0277ada19d0b4ba770b27a7d500c"),
    "math500_medium": ("main_test/math500_medium.jsonl", 105, "726a804a64c9cf63b948159fa73dc01c36f675d2e8e8d464083ec0fa3dc98aa0"),
    "mbpp_plus": ("main_test/mbpp_plus.jsonl", 378, "2a38ee610526fc27b45a1b67426f27b8fc9f4b134ac932ac4ed65b09f90166f6"),
    "medcalc": ("main_test/medcalc_bench_verified_test.jsonl", 1100, "10141832c4327477603d00b7f24bcb24e9892cb265051ea92162cca90845be7e"),
    "math500_high_level": ("hard_test/math500_high_level.jsonl", 262, "111ea34ab2c7ef4c5ef19735be9c1cbe3ea1a4dfcc77af2286c58aefb635a249"),
    "gsm8k": ("diagnostic_test/gsm8k.jsonl", 1319, "fb8df8e5d83a0f75e4cc781c46507d2d298c55c8d9c52e003c96583eaa20368f"),
    "mbpp_simple": ("diagnostic_test/mbpp_simple.jsonl", 257, "bcc197414b924ffea07cabe55d1cad80ad3a5428fe46600c49afadc839cec066"),
    "scienceqa": ("diagnostic_test/scienceqa.jsonl", 2224, "9606844d1fb78a0578216a1ae7d4b20b78907eb1a4cc4c8ceaffda4dd8f2f3be"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_paper_data(root: Path, allow_unverified: bool) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        if allow_unverified:
            return {"verified": False, "reason": "manifest.json absent"}
        raise FileNotFoundError(
            f"{manifest_path} is required; build it with prepare_standard11.py, "
            "or pass --allow-unverified-data only for an exploratory run"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strict = (
        manifest.get("contract") == "paper_11_task_suite"
        and manifest.get("strict_archived_reference") is True
        and manifest.get("all_tasks_match_archived_reference") is True
    )
    errors = []
    tasks = manifest.get("tasks", {})
    for task, (relative, rows, digest) in PAPER_REFERENCE.items():
        entry = tasks.get(task, {})
        path = root / relative
        if entry.get("path") != relative or entry.get("rows") != rows or entry.get("sha256") != digest:
            errors.append(f"{task}: manifest contract mismatch")
            continue
        if not path.is_file() or file_sha256(path) != digest:
            errors.append(f"{task}: artifact absent or SHA256 mismatch")
    if (not strict or errors) and not allow_unverified:
        raise RuntimeError(
            "Evaluation data does not match the frozen paper suite: "
            + ("manifest is non-strict; " if not strict else "")
            + "; ".join(errors)
        )
    return {
        "verified": bool(strict and not errors),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "errors": errors,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return rows[:limit] if limit else rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def model_context_length(model) -> int:
    return int(getattr(model.config, "max_position_embeddings", 32768))


def eos_ids(tokenizer):
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    for token in ["<|im_end|>", "<|endoftext|>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
            ids.append(token_id)
    if not ids:
        return None
    return ids[0] if len(ids) == 1 else ids


def load_model(base_model_path: Path, adapter_path: Path | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.to("cuda")
    model.eval()
    return model, tokenizer


def render_prompt(tokenizer, text: str, mode: str) -> str:
    if mode == "chat" or (mode == "auto" and getattr(tokenizer, "chat_template", None)):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return text


def render_messages(tokenizer, messages: list[dict[str, str]], mode: str) -> str:
    if mode == "chat" or (mode == "auto" and getattr(tokenizer, "chat_template", None)):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    chunks = []
    for message in messages:
        role = str(message.get("role", "user")).capitalize()
        chunks.append(f"{role}:\n{message.get('content', '')}")
    chunks.append("Assistant:\n")
    return "\n\n".join(chunks)


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    outputs: list[str] = []
    max_context = model_context_length(model)
    stop_ids = eos_ids(tokenizer)
    do_sample = temperature > 0
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        input_len = int(encoded["input_ids"].shape[1])
        if input_len + max_new_tokens > max_context:
            keep = max_context - max_new_tokens
            if keep <= 0:
                raise ValueError(f"max_new_tokens={max_new_tokens} exceeds context={max_context}")
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=keep,
                add_special_tokens=False,
            )
            input_len = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_ids,
            )
        for index in range(len(chunk)):
            new_tokens = generated[index, input_len:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return outputs


def score_choices_batch(model, tokenizer, prompts: list[str], choices_list: list[list[str]], batch_size: int):
    max_context = model_context_length(model)
    pad_id = tokenizer.pad_token_id
    flat: list[tuple[int, int, list[int], int]] = []
    for question_index, (prompt, choices) in enumerate(zip(prompts, choices_list)):
        context_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for choice_index, choice in enumerate(choices):
            choice_ids = tokenizer(" " + str(choice), add_special_tokens=False)["input_ids"]
            if len(context_ids) + len(choice_ids) > max_context:
                keep = max_context - len(choice_ids)
                context_ids = context_ids[-max(keep, 1) :]
            flat.append((question_index, choice_index, list(context_ids) + list(choice_ids), len(context_ids)))
    scores: list[list[float]] = [[float("-inf")] * len(choices) for choices in choices_list]
    for start in range(0, len(flat), batch_size):
        chunk = flat[start : start + batch_size]
        max_len = max(len(item[2]) for item in chunk)
        ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device="cuda")
        attention = torch.zeros((len(chunk), max_len), dtype=torch.long, device="cuda")
        for index, (_, _, token_ids, _) in enumerate(chunk):
            ids[index, : len(token_ids)] = torch.tensor(token_ids, device="cuda")
            attention[index, : len(token_ids)] = 1
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=attention).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        for index, (question_index, choice_index, token_ids, continuation_start) in enumerate(chunk):
            continuation = ids[index, continuation_start : len(token_ids)]
            predicted = log_probs[index, continuation_start - 1 : len(token_ids) - 1]
            value = predicted.gather(-1, continuation.unsqueeze(-1)).squeeze(-1).mean()
            scores[question_index][choice_index] = float(value.item())
    return [(max(range(len(values)), key=lambda idx: values[idx]), values) for values in scores]


def accuracy(correct: int, total: int) -> dict[str, Any]:
    return {"n": total, "correct": correct, "accuracy": correct / max(total, 1)}


def answer_spans(text: str) -> list[str]:
    """Return likely final-answer spans from strongest to weakest evidence."""
    text = text or ""
    spans: list[str] = []
    spans.extend(reversed(balanced_command_contents(text, "boxed")))
    spans.extend(reversed(HASH_ANSWER.findall(text)))
    spans.extend(reversed([match.group(1) for match in ANSWER_IS.finditer(text)]))
    nonempty = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if nonempty:
        spans.append(nonempty[-1])
    spans.append(text)
    return spans


def balanced_command_contents(text: str, command: str) -> list[str]:
    """Extract LaTeX command bodies while preserving nested braces."""
    marker = f"\\{command}"
    out: list[str] = []
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            break
        brace = start + len(marker)
        while brace < len(text) and text[brace].isspace():
            brace += 1
        if brace >= len(text) or text[brace] != "{":
            cursor = start + len(marker)
            continue
        depth = 0
        for end in range(brace, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[brace + 1 : end])
                    cursor = end + 1
                    break
        else:
            cursor = brace + 1
    return out


def parse_number_text(raw: str) -> tuple[float | None, bool]:
    raw = str(raw).strip().strip(".。,:;，；")
    raw = raw.replace("$", "").replace(",", "").replace("\\,", "").replace("\\!", "")
    raw = raw.replace("\\left", "").replace("\\right", "")
    raw = raw.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    percent = raw.endswith("%")
    raw = raw.rstrip("%").strip()
    latex = LATEX_FRAC.fullmatch(raw)
    if latex:
        denom = float(latex.group(2))
        return (float(latex.group(1)) / denom if denom else None), percent
    simple = SLASH_FRAC.fullmatch(raw)
    if simple:
        try:
            frac = Fraction(raw.replace(" ", ""))
            return float(frac), percent
        except Exception:
            return None, percent
    try:
        return float(raw), percent
    except ValueError:
        return None, percent


def numeric_candidates(text: str) -> list[tuple[float, bool]]:
    out: list[tuple[float, bool]] = []
    for span in answer_spans(text):
        span = span.strip()
        fraction_candidates = [match.group(0) for match in LATEX_FRAC.finditer(span)]
        fraction_candidates.extend(match.group(0) for match in SLASH_FRAC.finditer(span))
        candidates = fraction_candidates if fraction_candidates else [match.group(0) for match in NUMBER.finditer(span)]
        parsed = [parse_number_text(item) for item in candidates]
        parsed = [(value, percent) for value, percent in parsed if value is not None]
        if parsed:
            out.append(parsed[-1])
            break
    return out


def first_number(text: str) -> tuple[float | None, bool]:
    values = numeric_candidates(text)
    return values[0] if values else (None, False)


def numeric_matches(prediction: str, target: str, lower: str | None = None, upper: str | None = None) -> bool:
    if lower is not None and upper is not None:
        try:
            pred_value, _ = first_number(prediction)
            return pred_value is not None and float(lower) <= pred_value <= float(upper)
        except ValueError:
            return str(target).strip().lower() in str(prediction).strip().lower()
    pred_value, pred_percent = first_number(prediction)
    target_value, target_percent = first_number(str(target))
    if pred_value is None or target_value is None:
        return False
    candidates = [pred_value]
    if target_percent and not pred_percent:
        candidates.append(pred_value * 100.0)
    if pred_percent and not target_percent:
        candidates.append(pred_value / 100.0)
    tolerance = max(abs(target_value) * 1e-6, 1e-4)
    return any(abs(value - target_value) <= tolerance for value in candidates)


def normalize_text_answer(text: str) -> str:
    text = str(text or "").strip()
    spans = answer_spans(text)
    text = spans[0] if spans and spans[0].strip() else text
    text = text.strip().strip(".。,:;，；")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def exact_text_answer_matches(prediction: str, target: str) -> bool:
    target_norm = normalize_text_answer(target)
    if not target_norm:
        return False
    pred_norm = normalize_text_answer(prediction)
    return pred_norm == target_norm or pred_norm.startswith(target_norm + " ")


def normalize_math_text(text: str) -> str:
    text = str(text)
    boxed = balanced_command_contents(text, "boxed")
    if boxed:
        text = boxed[0]
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace(" ", "").replace("$", "")
    text = text.replace("\\,", "").replace("\\!", "")
    return text.lower().strip().strip(".")


def math_answer_span(prediction: str) -> str:
    for span in answer_spans(prediction):
        if span.strip():
            return span.strip()
    return ""


def target_is_numeric_math(target: str) -> bool:
    target = str(target).strip()
    if LATEX_FRAC.search(target) or SLASH_FRAC.search(target):
        return True
    stripped = target.replace("$", "").replace(",", "").strip()
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?%?", stripped))


def math_matches(prediction: str, target: str) -> bool:
    candidate = math_answer_span(prediction)
    norm_target = normalize_math_text(target)
    norm_candidate = normalize_math_text(candidate)
    if norm_target and norm_target == norm_candidate:
        return True
    if target_is_numeric_math(target):
        return numeric_matches(candidate, target)
    try:
        from math_verify import parse, verify  # type: ignore

        expected = parse(f"\\boxed{{{target}}}")
        predicted = parse(f"\\boxed{{{candidate}}}")
        return bool(verify(expected, predicted))
    except Exception:
        return False


class CodeExtractionError(Exception):
    def __init__(self, message: str, code: str = "") -> None:
        super().__init__(message)
        self.code = code


def trim_parseable_code(
    code: str,
    max_chars: int = 0,
    max_lines: int = 0,
    raise_on_resource_error: bool = False,
) -> str:
    if max_chars and len(code) > max_chars:
        message = f"candidate code too long for safe parsing: chars={len(code)} max_chars={max_chars}"
        if raise_on_resource_error:
            raise CodeExtractionError(message, code[:max_chars])
        code = code[:max_chars]
    lines = code.splitlines()
    if max_lines and len(lines) > max_lines:
        message = f"candidate code has too many lines for safe parsing: lines={len(lines)} max_lines={max_lines}"
        if raise_on_resource_error:
            raise CodeExtractionError(message, "\n".join(lines[:max_lines]))
        lines = lines[:max_lines]
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        except (MemoryError, RecursionError) as exc:
            message = f"candidate code parse resource error: {type(exc).__name__}: {exc}"
            if raise_on_resource_error:
                raise CodeExtractionError(message, candidate)
            return code.strip()
        if re.search(r"\b(def|class|import|from)\s+", candidate) or "input(" in candidate or "sys.stdin" in candidate:
            return candidate
    return code.strip()


def extract_code(
    text: str,
    starter_code: str = "",
    fn_name: str | None = None,
    max_chars: int = 0,
    max_lines: int = 0,
    raise_on_resource_error: bool = False,
) -> str:
    fenced = [match.strip() for match in code_fences_preserve_indent(text) if match.strip()]
    if fenced:
        candidates = fenced
    else:
        lines = (text or "").splitlines()
        start = None
        for index, line in enumerate(lines):
            if re.match(r"\s*(from\s+\S+\s+import\s+|import\s+\S+|def\s+\w+|class\s+\w+|if\s+__name__|sys\.stdin|input\()", line):
                start = index
                break
        candidates = ["\n".join(lines[start:]).strip() if start is not None else (text or "").strip()]

    ranked: list[tuple[int, int, str]] = []
    for index, candidate in enumerate(candidates):
        code = candidate
        if fn_name and starter_code and not re.search(rf"\bdef\s+{re.escape(fn_name)}\s*\(", code):
            code = starter_code.rstrip() + "\n" + code
        code = trim_parseable_code(
            code,
            max_chars=max_chars,
            max_lines=max_lines,
            raise_on_resource_error=raise_on_resource_error,
        )
        score = 0
        try:
            tree = ast.parse(code)
            score += 100
            top_functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if top_functions:
                score += 20
            if fn_name and fn_name in top_functions:
                score += 100
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            pass
        # Later code blocks are normally the model's corrected/final answer.
        ranked.append((score, index, code))
    return max(ranked, key=lambda item: (item[0], item[1]))[2] if ranked else ""


def code_fences_preserve_indent(text: str) -> list[str]:
    return [
        match.group(1)
        for match in re.finditer(
            r"```(?:python|py)?[^\n\r]*\r?\n(.*?)```",
            text or "",
            re.DOTALL | re.IGNORECASE,
        )
    ]


def first_code_fence_preserve_indent(text: str) -> str | None:
    candidates = code_fences_preserve_indent(text)
    return candidates[0] if candidates else None


def trim_blank_lines_preserve_indent(code: str) -> str:
    lines = code.replace("\t", "    ").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def strip_humaneval_tail(code: str) -> str:
    lines = code.splitlines()
    cut = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"def\s+check\s*\(", stripped):
            cut = index
            break
        if stripped.startswith("### Canonical solution") or stripped.startswith("### Unit tests"):
            cut = index
            break
        if line == stripped and stripped.startswith("assert "):
            cut = index
            break
    return "\n".join(lines[:cut])


def prompt_imports(prompt: str) -> str:
    imports = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)
            continue
        if stripped:
            break
    return "\n".join(imports)


def has_top_level_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, MemoryError, RecursionError):
        return False
    return any(isinstance(node, ast.FunctionDef) for node in tree.body)


def indent_body_if_needed(body: str) -> str:
    lines = body.splitlines()
    nonblank = [line for line in lines if line.strip()]
    if not nonblank:
        return body
    if any(line.startswith((" ", "\t")) for line in nonblank):
        return body
    return "\n".join(("    " + line if line.strip() else line) for line in lines)


def compiles_as_module(code: str) -> bool:
    try:
        compile(code, "<humaneval_candidate>", "exec")
        return True
    except (SyntaxError, MemoryError, RecursionError):
        return False


def extract_humaneval_code(reply: str, prompt: str, entry_point: str) -> str:
    reply_candidates = code_fences_preserve_indent(reply)
    if not reply_candidates:
        reply_candidates = [reply or ""]
    imports = prompt_imports(prompt)
    ranked: list[tuple[int, int, str]] = []

    for index, candidate in enumerate(reply_candidates):
        raw = strip_humaneval_tail(trim_blank_lines_preserve_indent(candidate))
        full_candidate = trim_parseable_code(raw)
        if has_top_level_function(full_candidate):
            exact_name = bool(re.search(rf"(?m)^\s*def\s+{re.escape(entry_point)}\s*\(", full_candidate))
            if imports and not full_candidate.lstrip().startswith(("import ", "from ")):
                full_candidate = imports.rstrip() + "\n\n" + full_candidate
            full_candidate = rewrite_single_function_name(full_candidate, entry_point)
            if compiles_as_module(full_candidate):
                ranked.append((220 if exact_name else 200, index, full_candidate))

        assembled = prompt + raw
        if not compiles_as_module(assembled):
            assembled = prompt + indent_body_if_needed(raw)
        assembled = rewrite_single_function_name(assembled, entry_point)
        if compiles_as_module(assembled):
            ranked.append((100, index, assembled))

    if ranked:
        return max(ranked, key=lambda item: (item[0], item[1]))[2]

    raw = strip_humaneval_tail(trim_blank_lines_preserve_indent(reply_candidates[-1]))
    assembled = prompt + raw
    if not compiles_as_module(assembled):
        assembled = prompt + indent_body_if_needed(raw)
    return rewrite_single_function_name(assembled, entry_point)


def rewrite_single_function_name(code: str, expected_name: str | None) -> str:
    if not expected_name:
        return code
    try:
        tree = ast.parse(code)
    except (SyntaxError, MemoryError, RecursionError):
        return code
    top_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if any(node.name == expected_name for node in top_functions):
        return code
    if not top_functions:
        return code
    # Generated code often contains helpers plus a wrongly named public
    # function. In that common case the public function is usually last.
    target_function = top_functions[-1]
    old_name = target_function.name
    target_function.name = expected_name

    class RenameCalls(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):  # noqa: N802
            if node.id == old_name:
                return ast.copy_location(ast.Name(id=expected_name, ctx=node.ctx), node)
            return node

    tree = RenameCalls().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return code


def run_python(script: str, timeout_seconds: int = 8) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
        handle.write(script)
        path = Path(handle.name)
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return result.returncode == 0, (result.stdout + "\n" + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        path.unlink(missing_ok=True)


def build_io_tests(io_spec: dict[str, Any], max_tests: int) -> list[dict[str, Any]]:
    inputs = io_spec.get("inputs") or []
    outputs = io_spec.get("outputs") or []
    tests = [{"input": inp, "output": out} for inp, out in zip(inputs, outputs)]
    return tests[:max_tests] if max_tests and max_tests > 0 else tests


def evaluate_io_code(code: str, io_spec: dict[str, Any], timeout: float, max_tests: int) -> dict[str, Any]:
    tests = build_io_tests(io_spec, max_tests)
    if not tests:
        return {"passed": False, "status": "no_tests", "passed_tests": 0, "total_tests": 0}
    payload = {"code": code, "tests": tests, "fn_name": io_spec.get("fn_name")}
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
        handle.write(APPS_HARNESS)
        harness_path = Path(handle.name)
    try:
        result = subprocess.run(
            [sys.executable, str(harness_path)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        line = (result.stdout or "").strip().splitlines()[-1] if (result.stdout or "").strip() else ""
        if not line:
            return {"passed": False, "status": "harness_error", "passed_tests": 0, "total_tests": len(tests), "stderr": (result.stderr or "")[-500:]}
        return json.loads(line)
    except subprocess.TimeoutExpired:
        return {"passed": False, "status": "timeout", "passed_tests": 0, "total_tests": len(tests)}
    finally:
        harness_path.unlink(missing_ok=True)


def metric_key(metric: dict[str, Any]) -> float | None:
    return metric.get("accuracy", metric.get("pass_at_1", metric.get("score")))


def recompute_means(result: dict[str, Any]) -> None:
    result.setdefault("main", {})
    result.setdefault("hard", {})
    result.setdefault("diagnostic", {})
    main_scoreable = [task for task in MAIN_TASKS if task not in UNSCORED_GENERATION_TASKS and task in result["main"]]
    if main_scoreable:
        result["main"]["main_scoreable_mean"] = sum(metric_key(result["main"][task]) for task in main_scoreable) / len(main_scoreable)
    if all(task in result["hard"] for task in HARD_TASKS):
        result["hard"]["hard_mean"] = sum(metric_key(result["hard"][task]) for task in HARD_TASKS) / len(HARD_TASKS)
    if all(task in result["diagnostic"] for task in DIAGNOSTIC_TASKS):
        result["diagnostic"]["diagnostic_mean"] = sum(metric_key(result["diagnostic"][task]) for task in DIAGNOSTIC_TASKS) / len(DIAGNOSTIC_TASKS)
    all_values = []
    completed_tasks = []
    for task in PAPER_SCOREABLE_TASKS:
        metric = result.get(task_suite(task), {}).get(task)
        if isinstance(metric, dict):
            value = metric_key(metric)
            if value is not None:
                all_values.append(value)
                completed_tasks.append(task)
    if all_values:
        result["scoreable_overall_mean"] = sum(all_values) / len(all_values)
        result["scoreable_dataset_count"] = len(all_values)
        result["scoreable_dataset_names"] = completed_tasks
        result["scoreable_standard"] = "paper_11_tasks_no_apps_no_health_no_planbench"


def task_suite(task: str) -> str:
    if task in MAIN_TASKS:
        return "main"
    if task in HARD_TASKS:
        return "hard"
    if task in DIAGNOSTIC_TASKS:
        return "diagnostic"
    raise ValueError(task)


def run_task(task: str, fn, result: dict[str, Any], metrics_path: Path, resume: bool) -> None:
    suite = task_suite(task)
    result.setdefault(suite, {})
    if resume and task in result[suite]:
        print(json.dumps({"skipped": f"{suite}.{task}", "reason": "already completed"}, ensure_ascii=False), flush=True)
        return
    started = time.time()
    result[suite][task] = fn()
    result.setdefault("suite_wall_seconds", {})[f"{suite}.{task}"] = round(time.time() - started, 1)
    recompute_means(result)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": f"{suite}.{task}", "wall_seconds": result["suite_wall_seconds"][f"{suite}.{task}"]}, ensure_ascii=False), flush=True)
    torch.cuda.empty_cache()


def evaluate_arc(model, tokenizer, args) -> dict[str, Any]:
    rows = read_jsonl(MAIN / "arc_challenge.jsonl", args.limit)
    prompts, choices = [], []
    for row in rows:
        rendered = "\n".join(f"{label}. {text}" for label, text in zip(row["choice_labels"], row["choice_texts"]))
        prompts.append(render_prompt(tokenizer, f"Answer this science question.\n\n{row['question']}\n{rendered}\n\nAnswer:", args.prompt_mode))
        choices.append(row["choice_texts"])
    preds = score_choices_batch(model, tokenizer, prompts, choices, args.choice_batch_size)
    records = []
    correct = 0
    for (idx, scores), row in zip(preds, rows):
        predicted = row["choice_labels"][idx]
        ok = predicted == row["answer"]
        correct += int(ok)
        records.append(
            {
                "id": row.get("id"),
                "question": row["question"],
                "choice_labels": row["choice_labels"],
                "choice_texts": row["choice_texts"],
                "choice_scores": scores,
                "predicted_answer": predicted,
                "target": row["answer"],
                "passed": bool(ok),
            }
        )
    write_jsonl(args.output_dir / "arc_challenge_results.jsonl", records)
    return accuracy(correct, len(rows))


def evaluate_scienceqa(model, tokenizer, args) -> dict[str, Any]:
    rows = read_jsonl(DIAG / "scienceqa.jsonl", args.limit)
    prompts, choices = [], []
    for row in rows:
        rendered = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(row["choices"]))
        hint = f"\nHint: {row['hint']}" if row.get("hint") else ""
        prompts.append(render_prompt(tokenizer, f"Answer the science question.{hint}\n\n{row['question']}\n{rendered}\n\nAnswer:", args.prompt_mode))
        choices.append(row["choices"])
    preds = score_choices_batch(model, tokenizer, prompts, choices, args.choice_batch_size)
    records = []
    correct = 0
    for (idx, scores), row in zip(preds, rows):
        target = int(row["answer"])
        ok = idx == target
        correct += int(ok)
        records.append(
            {
                "id": row.get("id"),
                "question": row["question"],
                "choices": row["choices"],
                "choice_scores": scores,
                "predicted_index": idx,
                "predicted_answer": row["choices"][idx],
                "target_index": target,
                "target": row["choices"][target],
                "passed": bool(ok),
            }
        )
    write_jsonl(args.output_dir / "scienceqa_results.jsonl", records)
    return accuracy(correct, len(rows))


def evaluate_gsm8k(model, tokenizer, args) -> dict[str, Any]:
    rows = read_jsonl(DIAG / "gsm8k.jsonl", args.limit)
    if args.limit == 0 and len(rows) != GSM8K_EXPECTED_ROWS:
        raise ValueError(
            f"Formal GSM8K evaluation requires the full official test split: "
            f"expected {GSM8K_EXPECTED_ROWS} rows, found {len(rows)}. "
            "Use --limit only for explicitly labelled smoke tests."
        )
    prompts = [
        render_prompt(
            tokenizer,
            "Solve the problem step by step and end with 'The answer is N.'\n\n" + GSM_SHOTS + f"Q: {row['question']}\nA:",
            args.prompt_mode,
        )
        for row in rows
    ]
    replies = generate_batch(model, tokenizer, prompts, 256, args.generation_batch_size, args.temperature, args.top_p)
    correct = 0
    records = []
    for reply, row in zip(replies, rows):
        scored_reply = re.split(r"\n\s*(?:Q:|Question:)", reply, maxsplit=1)[0]
        ok = numeric_matches(scored_reply, str(row["answer"]))
        correct += int(ok)
        records.append(
            {
                "id": row.get("id"),
                "generated_text": reply,
                "answer_span": answer_spans(scored_reply)[0] if answer_spans(scored_reply) else "",
                "target": str(row["answer"]),
                "passed": bool(ok),
            }
        )
    write_jsonl(args.output_dir / "gsm8k_results.jsonl", records)
    result = accuracy(correct, len(rows))
    result.update(
        {
            "dataset_variant": "openai_gsm8k_official_test_full" if args.limit == 0 else "explicit_debug_subset",
            "split": "test",
            "source_revision": GSM8K_SOURCE_REVISION,
            "source_sha256": GSM8K_SOURCE_SHA256,
            "formal_result": args.limit == 0 and len(rows) == GSM8K_EXPECTED_ROWS,
        }
    )
    return result


def evaluate_math_file(model, tokenizer, args, path: Path, max_new_tokens: int = 512) -> dict[str, Any]:
    rows = read_jsonl(path, args.limit)
    prompts = [
        render_prompt(tokenizer, "Solve the math problem. Put the final answer in \\boxed{}.\n\nProblem: " + row["problem"], args.prompt_mode)
        for row in rows
    ]
    replies = generate_batch(model, tokenizer, prompts, max_new_tokens, args.generation_batch_size, args.temperature, args.top_p)
    records = []
    correct = 0
    for reply, row in zip(replies, rows):
        ok = math_matches(reply, row["answer"])
        correct += int(ok)
        records.append(
            {
                "id": row.get("id"),
                "generated_text": reply,
                "answer_span": math_answer_span(reply),
                "target": str(row["answer"]),
                "passed": bool(ok),
            }
        )
    write_jsonl(args.output_dir / f"{path.stem}_results.jsonl", records)
    return accuracy(correct, len(rows))


def evaluate_numeric_file(model, tokenizer, args, path: Path, interval: bool) -> dict[str, Any]:
    rows = read_jsonl(path, args.limit)
    prompts = [render_prompt(tokenizer, row["prompt"], args.prompt_mode) for row in rows]
    replies = generate_batch(model, tokenizer, prompts, 128, args.generation_batch_size, args.temperature, args.top_p)
    correct = 0
    records = []
    for row, reply in zip(rows, replies):
        scoring = row.get("scoring", {})
        ok = numeric_matches(
            reply,
            row["target"],
            scoring.get("lower_limit") if interval else None,
            scoring.get("upper_limit") if interval else None,
        )
        correct += int(ok)
        records.append(
            {
                "id": row.get("id"),
                "generated_text": reply,
                "answer_span": answer_spans(reply)[0] if answer_spans(reply) else "",
                "target": str(row["target"]),
                "lower_limit": scoring.get("lower_limit") if interval else None,
                "upper_limit": scoring.get("upper_limit") if interval else None,
                "passed": bool(ok),
            }
        )
    write_jsonl(args.output_dir / f"{path.stem}_results.jsonl", records)
    return accuracy(correct, len(rows))


def evaluate_finqa_file(model, tokenizer, args) -> dict[str, Any]:
    rows = read_jsonl(MAIN / "finqa_official_test.jsonl", args.limit)
    prompts = [render_prompt(tokenizer, row["prompt"], args.prompt_mode) for row in rows]
    replies = generate_batch(model, tokenizer, prompts, 128, args.generation_batch_size, args.temperature, args.top_p)
    correct = 0
    scored = 0
    skipped_empty_target = 0
    text_targets = 0
    numeric_targets = 0
    records = []
    for row, reply in zip(rows, replies):
        target = str(row.get("target", ""))
        if not target.strip():
            skipped_empty_target += 1
            records.append(
                {
                    "id": row.get("id"),
                    "generated_text": reply,
                    "target": target,
                    "passed": None,
                    "status": "skipped_empty_target",
                }
            )
            continue
        if first_number(target)[0] is None:
            text_targets += 1
            ok = exact_text_answer_matches(reply, target)
            target_type = "text"
        else:
            numeric_targets += 1
            ok = numeric_matches(reply, target)
            target_type = "numeric"
        scored += 1
        correct += int(ok)
        records.append(
            {
                "id": row.get("id"),
                "generated_text": reply,
                "answer_span": answer_spans(reply)[0] if answer_spans(reply) else "",
                "target": target,
                "target_type": target_type,
                "passed": bool(ok),
                "status": "passed" if ok else "answer_mismatch",
            }
        )
    write_jsonl(args.output_dir / "finqa_official_test_results.jsonl", records)
    result = accuracy(correct, scored)
    result["raw_n"] = len(rows)
    result["skipped_empty_target"] = skipped_empty_target
    result["numeric_targets"] = numeric_targets
    result["text_targets"] = text_targets
    return result


def evaluate_legalbench(model, tokenizer, args) -> dict[str, Any]:
    rows = read_jsonl(MAIN / "legalbench_rule_application_test.jsonl", args.limit)
    labels_by_task: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["target"] not in labels_by_task[row["task_type"]]:
            labels_by_task[row["task_type"]].append(row["target"])
    prompts = [row["prompt"] for row in rows]
    choices = [labels_by_task[row["task_type"]] for row in rows]
    preds = score_choices_batch(model, tokenizer, prompts, choices, args.choice_batch_size)
    records = []
    correct = 0
    for (idx, scores), choice_set, row in zip(preds, choices, rows):
        predicted = choice_set[idx]
        ok = predicted.lower() == row["target"].lower()
        correct += int(ok)
        records.append(
            {
                "id": row.get("id"),
                "task_type": row["task_type"],
                "prompt": row["prompt"],
                "choices": choice_set,
                "choice_scores": scores,
                "predicted_answer": predicted,
                "target": row["target"],
                "passed": bool(ok),
            }
        )
    write_jsonl(args.output_dir / "legalbench_results.jsonl", records)
    return accuracy(correct, len(rows))


def evaluate_humaneval(model, tokenizer, args) -> dict[str, Any]:
    rows = read_jsonl(MAIN / "humaneval.jsonl", args.limit)
    prompts = [
        render_prompt(tokenizer, "Complete the Python function. Output only executable Python code.\n\n" + row["prompt"], args.prompt_mode)
        for row in rows
    ]
    replies = generate_batch(model, tokenizer, prompts, args.max_new_tokens_code, args.generation_batch_size, args.temperature, args.top_p)

    def check(pair) -> dict[str, Any]:
        reply, row = pair
        full_code = ""
        try:
            full_code = extract_humaneval_code(reply, row["prompt"], row["entry_point"])
            ast.parse(full_code)
            ok, output = run_python(full_code + "\n\n" + row.get("test", "") + f"\n\ncheck({row['entry_point']})\n")
            status = "passed" if ok else ("timeout" if output == "timeout" else "test_or_runtime_failure")
        except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
            ok = False
            output = f"{type(exc).__name__}: {exc}"
            status = "code_parse_failure"
        except Exception as exc:
            ok = False
            output = f"{type(exc).__name__}: {exc}"
            status = "code_extraction_failure"
        return {
            "id": row.get("id"),
            "entry_point": row["entry_point"],
            "generated_text": reply,
            "extracted_code": full_code,
            "passed": bool(ok),
            "status": status,
            "error": output[:4000] if not ok else "",
        }

    with ThreadPoolExecutor(max_workers=16) as executor:
        records = list(executor.map(check, zip(replies, rows)))
    write_jsonl(args.output_dir / "humaneval_results.jsonl", records)
    correct = sum(int(record["passed"]) for record in records)
    return {"n": len(rows), "correct": correct, "pass_at_1": correct / max(len(rows), 1)}


def mbpp_entry_point(row: dict[str, Any]) -> str | None:
    skipped = {
        "abs", "all", "any", "bool", "dict", "float", "int", "len", "list", "max", "min",
        "round", "set", "sorted", "str", "sum", "tuple", "math.isclose", "np.allclose",
        "numpy.allclose", "isinstance", "range", "print",
    }

    def call_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    tests = "\n".join(row.get("test_list") or [])
    for test in row.get("test_list") or []:
        try:
            tree = ast.parse(test)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = call_name(node.func)
            if not name:
                continue
            root = name.split(".", 1)[0]
            if name in skipped or root in {"math", "np", "numpy"} or root in skipped:
                continue
            return name
    match = re.search(r"assert\s+(?:\w+\.)*([A-Za-z_]\w*)\s*\(", tests)
    if match and match.group(1) not in skipped:
        return match.group(1)
    try:
        tree = ast.parse(row.get("canonical_solution") or "")
        funcs = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        if funcs:
            return funcs[-1]
    except SyntaxError:
        pass
    return None


def render_mbpp_prompt(tokenizer, row: dict[str, Any], args) -> str:
    tests = "\n".join(row.get("test_list") or [])
    entry = mbpp_entry_point(row)
    name_hint = f" Implement the function named `{entry}`." if entry else ""
    text = (
        "Write a complete Python function for the task below."
        f"{name_hint} Output only executable Python code.\n\n"
        f"Task: {row['prompt']}\n\n"
        "The solution must pass these tests:\n"
        f"{tests}\n\n"
        "Python code:\n"
    )
    return render_prompt(tokenizer, text, args.prompt_mode)


def evaluate_mbpp_file(model, tokenizer, args, path: Path, use_plus: bool) -> dict[str, Any]:
    rows = read_jsonl(path, args.limit)
    prompts = [render_mbpp_prompt(tokenizer, row, args) for row in rows]
    replies = generate_batch(model, tokenizer, prompts, args.max_new_tokens_code, args.generation_batch_size, args.temperature, args.top_p)

    def check(pair) -> dict[str, Any]:
        reply, row = pair
        code = ""
        entry_point = None
        try:
            entry_point = mbpp_entry_point(row)
            code = extract_code(reply, fn_name=entry_point)
            code = rewrite_single_function_name(code, entry_point)
            imports = "\n".join(row.get("test_imports") or [])
            tests = "\n".join(row.get("test_list") or [])
            plus_test = row.get("plus_test") or ""
            script = imports + "\n\n" + code + "\n\n" + tests + "\n"
            if use_plus and plus_test:
                script += "\n" + plus_test + "\n"
            ast.parse(code)
            ok, output = run_python(script, timeout_seconds=12)
            status = "passed" if ok else ("timeout" if output == "timeout" else "test_or_runtime_failure")
        except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
            ok = False
            output = f"{type(exc).__name__}: {exc}"
            status = "code_parse_failure"
        except Exception as exc:
            ok = False
            output = f"{type(exc).__name__}: {exc}"
            status = "code_extraction_failure"
        return {
            "id": row.get("id"),
            "entry_point": entry_point,
            "generated_text": reply,
            "extracted_code": code,
            "passed": bool(ok),
            "status": status,
            "error": output[:4000] if not ok else "",
        }

    with ThreadPoolExecutor(max_workers=16) as executor:
        records = list(executor.map(check, zip(replies, rows)))
    write_jsonl(args.output_dir / f"{path.stem}_results.jsonl", records)
    correct = sum(int(record["passed"]) for record in records)
    return {"n": len(rows), "correct": correct, "pass_at_1": correct / max(len(rows), 1)}


def render_apps_prompt(tokenizer, row: dict[str, Any], args) -> str:
    text = (
        "Write a complete Python 3 program that solves the programming problem.\n"
        "Read input from standard input and write output to standard output.\n"
        "Return only valid Python code, with no explanation.\n\n"
        "Problem:\n"
        + row["prompt"]
        + "\n\nPython code:\n"
    )
    return render_prompt(tokenizer, text, args.prompt_mode)


def parse_input_output(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value or "{}")
    except Exception:
        return {}


def apps_failure_result(status: str, error: str, io_spec: dict[str, Any], max_tests: int) -> dict[str, Any]:
    try:
        total_tests = len(build_io_tests(io_spec, max_tests))
    except Exception:
        total_tests = 0
    return {
        "passed": False,
        "status": status,
        "passed_tests": 0,
        "total_tests": total_tests,
        "error": error[:1000],
    }


def evaluate_apps_hard(model, tokenizer, args) -> dict[str, Any]:
    rows = read_jsonl(HARD / "apps_hard.jsonl", args.limit)
    results_path = args.output_dir / "apps_hard_results.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if args.resume and results_path.exists():
        for line in results_path.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
    pending = [row for row in rows if row["id"] not in done]
    with results_path.open("a" if done else "w", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.apps_batch_size):
            batch_rows = pending[start : start + args.apps_batch_size]
            prompts = [render_apps_prompt(tokenizer, row, args) for row in batch_rows]
            replies = generate_batch(model, tokenizer, prompts, args.max_new_tokens_apps, args.apps_batch_size, args.temperature, args.top_p)
            for row, reply in zip(batch_rows, replies):
                io_spec = parse_input_output(row.get("input_output"))
                code = ""
                try:
                    code = extract_code(
                        reply,
                        row.get("starter_code") or "",
                        io_spec.get("fn_name"),
                        max_chars=args.apps_code_max_chars,
                        max_lines=args.apps_code_max_lines,
                        raise_on_resource_error=True,
                    )
                    eval_result = evaluate_io_code(code, io_spec, args.apps_exec_timeout, args.apps_max_tests_per_problem)
                except CodeExtractionError as exc:
                    code = exc.code
                    eval_result = apps_failure_result("code_extraction_error", str(exc), io_spec, args.apps_max_tests_per_problem)
                except (MemoryError, RecursionError) as exc:
                    eval_result = apps_failure_result(
                        "code_extraction_error",
                        f"{type(exc).__name__}: {exc}",
                        io_spec,
                        args.apps_max_tests_per_problem,
                    )
                except Exception as exc:
                    eval_result = apps_failure_result(
                        "apps_eval_error",
                        f"{type(exc).__name__}: {exc}",
                        io_spec,
                        args.apps_max_tests_per_problem,
                    )
                rec = {
                    "id": row["id"],
                    "problem_id": row.get("problem_id"),
                    "difficulty": row.get("difficulty"),
                    "url": row.get("url"),
                    "generated_text": reply,
                    "code": code[: args.apps_code_record_chars],
                    **eval_result,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[apps_hard] {args.model_name} {len(done) + start + 1}/{len(rows)} {row['id']} {rec['status']} {rec['passed']}", flush=True)
    rows_out = []
    for line in results_path.open(encoding="utf-8"):
        if line.strip():
            rows_out.append(json.loads(line))
    correct = sum(1 for row in rows_out if row.get("passed"))
    status_counts: dict[str, int] = {}
    for row in rows_out:
        status_counts[row.get("status", "unknown")] = status_counts.get(row.get("status", "unknown"), 0) + 1
    return {"n": len(rows), "correct": correct, "pass_at_1": correct / max(len(rows), 1), "status_counts": status_counts}


def health_messages_and_rubrics(task: str, row: dict[str, Any]) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    if task == "healthbench":
        rid = str(row.get("prompt_id"))
        messages = [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]
        rubrics = row["rubrics"]
        return rid, messages, rubrics
    rid = str(row.get("id"))
    messages = [{"role": m["role"], "content": m["content"]} for m in row["conversation"]["messages"]]
    rubrics = [{"criterion": item["criterion_text"], "points": item["points"], "tags": []} for item in row["rubric_items"]]
    return rid, messages, rubrics


def evaluate_healthbench_generation(model, tokenizer, args, task: str) -> dict[str, Any]:
    if args.skip_healthbench_generation:
        return {"n": 0, "status": "skipped_by_config", "score": None}
    path = MAIN / f"{task}.jsonl"
    rows = read_jsonl(path, args.limit)
    responses_path = args.output_dir / f"{task}_responses.jsonl"
    done: set[str] = set()
    if args.resume and responses_path.exists():
        for line in responses_path.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["id"])
    pending = []
    for index, row in enumerate(rows):
        rid, messages, rubrics = health_messages_and_rubrics(task, row)
        if rid in done:
            continue
        pending.append((index, rid, messages, rubrics, row))
    with responses_path.open("a" if done else "w", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.health_batch_size):
            chunk = pending[start : start + args.health_batch_size]
            prompts = [render_messages(tokenizer, item[2], args.health_prompt_mode) for item in chunk]
            replies = generate_batch(model, tokenizer, prompts, args.max_new_tokens_health, args.health_batch_size, args.health_temperature, args.health_top_p)
            for (index, rid, messages, rubrics, row), reply in zip(chunk, replies):
                rec = {
                    "dataset": task,
                    "model_name": args.model_name,
                    "seed": args.seed,
                    "id": rid,
                    "row_index": index,
                    "messages": messages,
                    "rubrics": rubrics,
                    "response": reply,
                    "metadata": {k: row.get(k) for k in ["example_tags", "use_case", "type", "difficulty", "specialty"] if k in row},
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[healthbench] {args.model_name} {task} {min(start + len(chunk), len(pending))}/{len(pending)}", flush=True)
    generated = sum(1 for line in responses_path.open(encoding="utf-8") if line.strip())
    return {
        "n": len(rows),
        "generated": generated,
        "score": None,
        "status": "responses_generated_ungraded",
        "reason": "HealthBench requires rubric grading with an external judge; this evaluator only generated model responses.",
        "responses_path": str(responses_path),
    }


def selected_tasks(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return MAIN_TASKS + HARD_TASKS + DIAGNOSTIC_TASKS
    out = []
    aliases = {
        "main": MAIN_TASKS,
        "main_test": MAIN_TASKS,
        "hard": HARD_TASKS,
        "hard_test": HARD_TASKS,
        "diagnostic": DIAGNOSTIC_TASKS,
        "diagnostic_test": DIAGNOSTIC_TASKS,
        "diag": DIAGNOSTIC_TASKS,
        "scoreable": SCOREABLE_TASKS,
        "all_scoreable": SCOREABLE_TASKS,
        "no_healthbench": SCOREABLE_TASKS,
    }
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        lower = key.lower()
        if lower in aliases:
            out.extend(aliases[lower])
        elif lower == "planbench_hard_all_tasks":
            continue
        elif lower in MAIN_TASKS + HARD_TASKS + DIAGNOSTIC_TASKS:
            out.append(lower)
        else:
            raise ValueError(f"Unknown task: {key}")
    return list(dict.fromkeys(out))


def main() -> None:
    global DATA_TEST, MAIN, HARD, DIAG
    args = parse_args()
    DATA_TEST = args.data_root.expanduser().resolve()
    MAIN = DATA_TEST / "main_test"
    HARD = DATA_TEST / "hard_test"
    DIAG = DATA_TEST / "diagnostic_test"
    if not DATA_TEST.is_dir():
        raise FileNotFoundError(f"external evaluation data root does not exist: {DATA_TEST}")
    requested_tasks = selected_tasks(args.tasks)
    data_contract = validate_paper_data(DATA_TEST, args.allow_unverified_data)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    if args.resume and metrics_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected_resume_contract = {
            "model_name": args.model_name,
            "base_model_path": str(args.base_model_path.expanduser().resolve()),
            "adapter_path": str(args.adapter_path.expanduser().resolve()) if args.adapter_path else None,
            "data_root": str(DATA_TEST),
            "seed": args.seed,
            "prompt_mode": args.prompt_mode,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "tasks_requested": requested_tasks,
        }
        mismatches = {
            key: {"stored": result.get(key), "requested": value}
            for key, value in expected_resume_contract.items()
            if result.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Refusing to resume metrics from a different evaluation contract: "
                + json.dumps(mismatches, ensure_ascii=False)
            )
    else:
        result = {
            "model_name": args.model_name,
            "base_model_path": str(args.base_model_path.expanduser().resolve()),
            "adapter_path": str(args.adapter_path.expanduser().resolve()) if args.adapter_path else None,
            "data_root": str(DATA_TEST),
            "data_contract": data_contract,
            "seed": args.seed,
            "prompt_mode": args.prompt_mode,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "tasks_requested": requested_tasks,
            "limit": args.limit,
            "choice_batch_size": args.choice_batch_size,
            "generation_batch_size": args.generation_batch_size,
            "max_new_tokens_code": args.max_new_tokens_code,
            "task_max_new_tokens": {
                "gsm8k": 256,
                "math500_medium": 512,
                "math500_high_level": 512,
                "finqa_medcalc": 128,
            },
            "health_prompt_mode": args.health_prompt_mode,
            "health_temperature": args.health_temperature,
            "health_top_p": args.health_top_p,
            "enable_thinking": False,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "paper_scoreable_standard": "11 tasks across code generation, mathematics, finance, medical, science QA and legal",
            "excluded_from_paper_aggregate": {
                "apps_hard": "excluded by user request",
                "healthbench": "excluded by user request",
                "healthbench_professional": "excluded by user request",
                "planbench_hard_all_tasks": "excluded by user request",
            },
        }

    result["data_contract"] = data_contract

    model, tokenizer = load_model(args.base_model_path, args.adapter_path)
    task_fns = {
        "arc_challenge": lambda: evaluate_arc(model, tokenizer, args),
        "finqa": lambda: evaluate_finqa_file(model, tokenizer, args),
        "healthbench": lambda: evaluate_healthbench_generation(model, tokenizer, args, "healthbench"),
        "healthbench_professional": lambda: evaluate_healthbench_generation(model, tokenizer, args, "healthbench_professional"),
        "humaneval": lambda: evaluate_humaneval(model, tokenizer, args),
        "legalbench": lambda: evaluate_legalbench(model, tokenizer, args),
        "math500_medium": lambda: evaluate_math_file(model, tokenizer, args, MAIN / "math500_medium.jsonl"),
        "mbpp_plus": lambda: evaluate_mbpp_file(model, tokenizer, args, MAIN / "mbpp_plus.jsonl", True),
        "medcalc": lambda: evaluate_numeric_file(model, tokenizer, args, MAIN / "medcalc_bench_verified_test.jsonl", True),
        "apps_hard": lambda: evaluate_apps_hard(model, tokenizer, args),
        "math500_high_level": lambda: evaluate_math_file(model, tokenizer, args, HARD / "math500_high_level.jsonl"),
        "gsm8k": lambda: evaluate_gsm8k(model, tokenizer, args),
        "mbpp_simple": lambda: evaluate_mbpp_file(model, tokenizer, args, DIAG / "mbpp_simple.jsonl", False),
        "scienceqa": lambda: evaluate_scienceqa(model, tokenizer, args),
    }
    for task in requested_tasks:
        run_task(task, task_fns[task], result, metrics_path, args.resume)
    result["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    recompute_means(result)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_name": args.model_name, "metrics": str(metrics_path), "scoreable_overall_mean": result.get("scoreable_overall_mean")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
