#!/usr/bin/env python3
"""Sandboxed strict all-tests-pass reward for the verified KodCode RL-10K subset."""

from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_BASE_PATH = Path(__file__).with_name("stage3_reward_base.py")
_BASE_SPEC = importlib.util.spec_from_file_location("diagnose_dapo_stage3_reward_base", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError(f"cannot load reward base: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)
_kill_process_group = _BASE._kill_process_group
_set_eval_limits = _BASE._set_eval_limits
extract_code = _BASE.extract_code


STATUS = {
    "passed": 0,
    "wrong_answer": 1,
    "syntax_error": 2,
    "runtime_error": 3,
    "timeout": 4,
    "sandbox_error": 5,
}

SANDBOX = r'''
import contextlib
import io
import json
import signal
import sys
import types

payload = json.loads(sys.stdin.read())
code = payload["code"]
test_code = payload["test_code"]
test_names = payload["test_functions"]
per_test_timeout = float(payload["per_test_timeout"])

def emit(status, passed_tests, total_tests, **extra):
    result = {
        "status": status,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        **extra,
    }
    sys.__stdout__.write(json.dumps(result, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()

def on_timeout(signum, frame):
    raise TimeoutError("per-test timeout")

total = len(test_names)
try:
    module = types.ModuleType("solution")
    module.__dict__["__name__"] = "solution"
    captured_out, captured_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
        exec(compile(code, "<generated_solution>", "exec"), module.__dict__)
    sys.modules["solution"] = module
    test_ns = dict(module.__dict__)
    test_ns["__name__"] = "kodcode_hidden_tests"
    with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
        exec(compile(test_code, "<hidden_tests>", "exec"), test_ns)
except SyntaxError as exc:
    emit("syntax_error", 0, total, error=str(exc)[:500])
    raise SystemExit
except BaseException as exc:
    emit("runtime_error", 0, total, error=f"{type(exc).__name__}: {exc}"[:500])
    raise SystemExit

signal.signal(signal.SIGALRM, on_timeout)
passed = 0
wrong = 0
runtime = 0
timeouts = 0
errors = []
for index, name in enumerate(test_names):
    function = test_ns.get(name)
    if not callable(function):
        runtime += 1
        errors.append({"test": name, "error": "missing test function"})
        continue
    try:
        signal.setitimer(signal.ITIMER_REAL, per_test_timeout)
        with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
            function()
        passed += 1
    except (AssertionError,) as exc:
        wrong += 1
        errors.append({"test": name, "error": f"AssertionError: {exc}"[:300]})
    except TimeoutError as exc:
        timeouts += 1
        errors.append({"test": name, "error": str(exc)})
    except BaseException as exc:
        runtime += 1
        errors.append({"test": name, "error": f"{type(exc).__name__}: {exc}"[:300]})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if timeouts >= 3:
        remaining = total - (index + 1)
        timeouts += remaining
        break

status = "passed" if passed == total else "timeout" if timeouts else "runtime_error" if runtime else "wrong_answer"
emit(
    status,
    passed,
    total,
    wrong_answers=wrong,
    runtime_errors=runtime,
    timeouts=timeouts,
    errors=errors[:5],
)
'''


def _ground_truth(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"unsupported ground truth type: {type(value).__name__}")


def _run(
    code: str,
    ground_truth: dict[str, Any],
    exec_timeout: float,
    per_test_timeout: float,
    memory_limit_gib: float,
    output_limit_mib: float,
) -> dict[str, Any]:
    payload = {
        "code": code,
        "test_code": ground_truth["test_code"],
        "test_functions": ground_truth["test_functions"],
        "per_test_timeout": per_test_timeout,
    }
    with tempfile.TemporaryDirectory(prefix="kodcode_reward_") as temp_dir:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", SANDBOX],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=temp_dir,
            start_new_session=True,
            preexec_fn=lambda: _set_eval_limits(
                memory_limit_gib, exec_timeout, output_limit_mib
            ),
        )
        try:
            stdout, stderr = proc.communicate(
                input=json.dumps(payload, ensure_ascii=False),
                timeout=exec_timeout,
            )
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            stdout, stderr = proc.communicate()
            return {
                "status": "timeout",
                "passed_tests": 0,
                "total_tests": len(ground_truth["test_functions"]),
                "error": "completion hard timeout",
            }
        finally:
            if proc.poll() is None:
                _kill_process_group(proc)
        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            return {
                "status": "sandbox_error",
                "passed_tests": 0,
                "total_tests": len(ground_truth["test_functions"]),
                "error": stderr[-500:],
            }
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return {
                "status": "sandbox_error",
                "passed_tests": 0,
                "total_tests": len(ground_truth["test_functions"]),
                "error": (stdout + "\n" + stderr)[-500:],
            }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    exec_timeout: float = 180.0,
    per_test_timeout: float = 1.0,
    memory_limit_gib: float = 2.0,
    output_limit_mib: float = 16.0,
    **_: Any,
) -> dict[str, float]:
    del data_source
    truth = _ground_truth(ground_truth)
    starter = ""
    code = extract_code(solution_str, starter)
    result = _run(
        code,
        truth,
        float(exec_timeout),
        float(per_test_timeout),
        float(memory_limit_gib),
        float(output_limit_mib),
    )
    passed_tests = int(result.get("passed_tests") or 0)
    total_tests = int(result.get("total_tests") or len(truth["test_functions"]))
    pass_frac = passed_tests / max(total_tests, 1)
    passed = total_tests > 0 and passed_tests == total_tests
    # Training reward is deliberately binary. pass_frac remains diagnostic only.
    score = 1.0 if passed else 0.0
    status = str(result.get("status") or "sandbox_error")
    return {
        "score": float(score),
        "passed": float(passed),
        "pass_frac": float(pass_frac),
        "status_code": float(STATUS.get(status, STATUS["sandbox_error"])),
        "passed_tests": float(passed_tests),
        "total_tests": float(total_tests),
        "wrong_answers": float(result.get("wrong_answers") or 0),
        "runtime_errors": float(result.get("runtime_errors") or 0),
        "timeouts": float(result.get("timeouts") or 0),
    }


if __name__ == "__main__":
    truth = {
        "test_code": (
            "from solution import add\n"
            "def test_one():\n    assert add(1, 2) == 3\n"
            "def test_two():\n    assert add(-1, 1) == 0\n"
        ),
        "test_functions": ["test_one", "test_two"],
    }
    good = compute_score("selftest", "def add(a, b):\n    return a + b\n", truth)
    bad = compute_score("selftest", "def add(a, b):\n    return 0\n", truth)
    assert good["score"] == 1.0 and good["passed"] == 1.0, good
    assert bad["score"] == 0.0 and bad["passed"] == 0.0, bad
    assert bad["pass_frac"] == 0.5, bad
    print(json.dumps({"good": good, "bad": bad}, indent=2))
