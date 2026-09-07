#!/usr/bin/env python3
"""Rule-based code reward for Stage 3 verl/DAPO smoke tests."""

from __future__ import annotations

import ast
import json
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
from typing import Any


FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
MAX_PARSE_CODE_CHARS = 65_536
MAX_PARSE_TRIM_ATTEMPTS = 64

STATUS_CODE = {
    "passed": 0,
    "syntax_error": 1,
    "missing_entry_point": 2,
    "entry_point_not_callable": 3,
    "wrong_answer": 4,
    "runtime_error": 5,
    "timeout": 6,
    "harness_error": 7,
    "no_tests": 8,
    "eval_error": 9,
}

EVAL_HARNESS = r'''
import ast
import io
import json
import math
import signal
import sys
import traceback
from bisect import *
from collections import *
from functools import *
from heapq import *
from itertools import *
from math import *
from operator import *
from string import *
from typing import *

payload = json.loads(sys.stdin.read())
code = payload["code"]
mode = payload["mode"]
entry_point = payload.get("entry_point")
tests = payload.get("tests") or []
per_test_timeout = float(payload.get("per_test_timeout", 2.0))

class FakeStdin:
    def __init__(self, text):
        self._text = io.StringIO(str(text))
        self.buffer = io.BytesIO(str(text).encode())
    def read(self, *args):
        return self._text.read(*args)
    def readline(self, *args):
        return self._text.readline(*args)
    def readlines(self, *args):
        return self._text.readlines(*args)
    def __iter__(self):
        return iter(self._text)

def norm_text(value):
    text = "" if value is None else str(value)
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()

def stdout_equal(got, expected):
    a, b = norm_text(got), norm_text(expected)
    return a == b or a.split() == b.split()

def value_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(value_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a.keys()) == set(b.keys()) and all(value_equal(a[k], b[k]) for k in a)
    return str(a) == str(b)

def result(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False))

def timeout_handler(signum, frame):
    raise TimeoutError("per-test timeout")

try:
    ast.parse(code)
except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
    result(
        passed=False,
        status="syntax_error",
        interface_ok=False,
        passed_tests=0,
        total_tests=len(tests),
        wrong_answers=0,
        runtime_errors=0,
        timeouts=0,
        error=str(exc)[:300],
    )
    raise SystemExit

try:
    if not tests:
        result(
            passed=False,
            status="no_tests",
            interface_ok=False,
            passed_tests=0,
            total_tests=0,
            wrong_answers=0,
            runtime_errors=0,
            timeouts=0,
        )
        raise SystemExit

    if mode == "function":
        ns = {"__name__": "__main__"}
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, per_test_timeout)
            exec(compile(code, "<solution>", "exec"), ns)
        except TimeoutError as exc:
            result(
                passed=False,
                status="timeout",
                interface_ok=False,
                passed_tests=0,
                total_tests=len(tests),
                wrong_answers=0,
                runtime_errors=0,
                timeouts=len(tests),
                error=str(exc)[:300],
            )
            raise SystemExit
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            candidate = eval(entry_point, ns)
        except Exception as exc:
            result(
                passed=False,
                status="missing_entry_point",
                interface_ok=False,
                passed_tests=0,
                total_tests=len(tests),
                wrong_answers=0,
                runtime_errors=0,
                timeouts=0,
                error=str(exc)[:300],
            )
            raise SystemExit
        if not callable(candidate):
            result(
                passed=False,
                status="entry_point_not_callable",
                interface_ok=False,
                passed_tests=0,
                total_tests=len(tests),
                wrong_answers=0,
                runtime_errors=0,
                timeouts=0,
            )
            raise SystemExit

        passed_tests = 0
        wrong_answers = 0
        runtime_errors = 0
        timeouts = 0
        for idx, test in enumerate(tests):
            args = test.get("args") or []
            kwargs = test.get("kwargs") or {}
            expected = test.get("output")
            try:
                signal.setitimer(signal.ITIMER_REAL, per_test_timeout)
                got = candidate(*args, **kwargs)
                if value_equal(got, expected):
                    passed_tests += 1
                else:
                    wrong_answers += 1
            except TimeoutError:
                timeouts += 1
            except Exception:
                runtime_errors += 1
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if timeouts >= 3:
                break
        skipped_after_timeout = len(tests) - (passed_tests + wrong_answers + runtime_errors + timeouts)
        timeouts += max(skipped_after_timeout, 0)
        passed = passed_tests == len(tests)
        status = "passed" if passed else "timeout" if timeouts else "runtime_error" if runtime_errors else "wrong_answer"
        result(
            passed=passed,
            status=status,
            interface_ok=True,
            passed_tests=passed_tests,
            total_tests=len(tests),
            wrong_answers=wrong_answers,
            runtime_errors=runtime_errors,
            timeouts=timeouts,
        )
    else:
        signal.signal(signal.SIGALRM, timeout_handler)
        passed_tests = 0
        wrong_answers = 0
        runtime_errors = 0
        timeouts = 0
        for idx, test in enumerate(tests):
            old_stdin, old_stdout = sys.stdin, sys.stdout
            fake_out = io.StringIO()
            ns = {"__name__": "__main__"}
            case_failed = False
            try:
                sys.stdin = FakeStdin(test.get("input", ""))
                sys.stdout = fake_out
                try:
                    signal.setitimer(signal.ITIMER_REAL, per_test_timeout)
                    exec(compile(code, "<solution>", "exec"), ns)
                except SystemExit:
                    pass
                except TimeoutError:
                    timeouts += 1
                    case_failed = True
                except Exception:
                    runtime_errors += 1
                    case_failed = True
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            finally:
                sys.stdin, sys.stdout = old_stdin, old_stdout
            if not case_failed:
                got = fake_out.getvalue()
                expected = test.get("output", "")
                if stdout_equal(got, expected):
                    passed_tests += 1
                else:
                    wrong_answers += 1
            if timeouts >= 3:
                break
        skipped_after_timeout = len(tests) - (passed_tests + wrong_answers + runtime_errors + timeouts)
        timeouts += max(skipped_after_timeout, 0)
        passed = passed_tests == len(tests)
        status = "passed" if passed else "timeout" if timeouts else "runtime_error" if runtime_errors else "wrong_answer"
        result(
            passed=passed,
            status=status,
            interface_ok=True,
            passed_tests=passed_tests,
            total_tests=len(tests),
            wrong_answers=wrong_answers,
            runtime_errors=runtime_errors,
            timeouts=timeouts,
        )
except TimeoutError as exc:
    result(
        passed=False,
        status="timeout",
        interface_ok=True,
        passed_tests=0,
        total_tests=len(tests),
        wrong_answers=0,
        runtime_errors=0,
        timeouts=len(tests),
        error=str(exc)[:300],
    )
except Exception as exc:
    result(
        passed=False,
        status="runtime_error",
        interface_ok=True,
        passed_tests=0,
        total_tests=len(tests),
        wrong_answers=0,
        runtime_errors=len(tests),
        timeouts=0,
        error=str(exc)[:300],
        traceback=traceback.format_exc()[-500:],
    )
'''


def _json_loads_maybe(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def _trim_parseable_code(code: str) -> str:
    # Generated text can contain NUL bytes, which make ast.parse raise
    # ValueError instead of SyntaxError. They are never valid Python source.
    text = code.replace("\x00", "").strip()[:MAX_PARSE_CODE_CHARS]
    if not text:
        return text
    lines = text.splitlines()
    first_end = max(1, len(lines) - MAX_PARSE_TRIM_ATTEMPTS + 1)
    for end in range(len(lines), first_end - 1, -1):
        candidate = "\n".join(lines[:end]).strip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
            return candidate
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            continue
    return text


def extract_code(text: str, starter_code: str = "") -> str:
    text = text or ""
    fenced = FENCE_RE.findall(text)
    if fenced:
        code = fenced[0].strip()
    else:
        lines = text.splitlines()
        start = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"(from\s+\S+\s+import\s+|import\s+\S+|class\s+\w+|def\s+\w+|if\s+__name__)", stripped):
                start = idx
                break
        code = "\n".join(lines[start:]).strip() if start is not None else text.strip()

    starter = (starter_code or "").strip()
    if starter and "class Solution" in code and starter not in code:
        code = starter.rstrip() + "\n\n" + code
    return _trim_parseable_code(code)


def _format_ok(solution_str: str) -> bool:
    text = (solution_str or "").strip()
    if not text:
        return False
    if "```" in text:
        return False
    prose_markers = ["Here is", "Explanation", "```python", "The solution"]
    return not any(marker.lower() in text[:160].lower() for marker in prose_markers)


def _set_eval_limits(memory_limit_gib: float, exec_timeout: float, output_limit_mib: float) -> None:
    memory_bytes = max(1, int(memory_limit_gib * 1024**3))
    output_bytes = max(1, int(output_limit_mib * 1024**2))
    cpu_seconds = max(1, int(exec_timeout) + 1)
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (output_bytes, output_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_eval(
    code: str,
    meta: dict[str, Any],
    exec_timeout: float,
    per_test_timeout: float,
    memory_limit_gib: float,
    output_limit_mib: float,
) -> dict[str, Any]:
    payload = {
        "code": code,
        "mode": meta.get("mode"),
        "entry_point": meta.get("entry_point"),
        "tests": meta.get("tests") or [],
        "per_test_timeout": per_test_timeout,
    }
    proc = None
    try:
        with (
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file,
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file,
        ):
            proc = subprocess.Popen(
                [sys.executable, "-c", EVAL_HARNESS],
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                start_new_session=True,
                preexec_fn=lambda: _set_eval_limits(memory_limit_gib, exec_timeout, output_limit_mib),
            )
            try:
                proc.communicate(input=json.dumps(payload, ensure_ascii=False), timeout=exec_timeout)
            except subprocess.TimeoutExpired:
                _kill_process_group(proc)
                proc.communicate()
                raise
            _kill_process_group(proc)
            stdout_file.seek(0)
            stdout = stdout_file.read()

        parsed = None
        for line in reversed(stdout.strip().splitlines()):
            candidate = _json_loads_maybe(line)
            if isinstance(candidate, dict) and "passed" in candidate and "total_tests" in candidate:
                parsed = candidate
                break
        if parsed is None:
            return {
                "passed": False,
                "status": "harness_error",
                "interface_ok": False,
                "passed_tests": 0,
                "total_tests": len(payload["tests"]),
                "wrong_answers": 0,
                "runtime_errors": len(payload["tests"]),
                "timeouts": 0,
            }
        return parsed
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "status": "timeout",
            "interface_ok": True,
            "passed_tests": 0,
            "total_tests": len(payload["tests"]),
            "wrong_answers": 0,
            "runtime_errors": 0,
            "timeouts": len(payload["tests"]),
        }
    except Exception:
        return {
            "passed": False,
            "status": "eval_error",
            "interface_ok": False,
            "passed_tests": 0,
            "total_tests": len(payload["tests"]),
            "wrong_answers": 0,
            "runtime_errors": len(payload["tests"]),
            "timeouts": 0,
        }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    exec_timeout: float = 6.0,
    per_test_timeout: float = 2.0,
    memory_limit_gib: float = 2.0,
    output_limit_mib: float = 16.0,
) -> dict[str, float]:
    """Return full-pass bonus or capped partial-credit execution reward."""
    del data_source
    meta = _json_loads_maybe(ground_truth)
    if not meta and extra_info:
        meta = _json_loads_maybe(extra_info)

    tests = meta.get("tests") or []
    format_ok = _format_ok(solution_str)
    try:
        code = extract_code(solution_str, meta.get("starter_code") or "")
        ast.parse(code)
        parse_ok = True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        code = ""
        parse_ok = False

    if not tests:
        eval_result = {
            "passed": False,
            "status": "no_tests",
            "interface_ok": False,
            "passed_tests": 0,
            "total_tests": 0,
        }
    elif not parse_ok:
        eval_result = {
            "passed": False,
            "status": "syntax_error",
            "interface_ok": False,
            "passed_tests": 0,
            "total_tests": len(tests),
        }
    else:
        try:
            eval_result = _run_eval(
                code,
                meta,
                exec_timeout=exec_timeout,
                per_test_timeout=per_test_timeout,
                memory_limit_gib=memory_limit_gib,
                output_limit_mib=output_limit_mib,
            )
        except (Exception, MemoryError) as exc:
            eval_result = {
                "passed": False,
                "status": "eval_error",
                "interface_ok": False,
                "passed_tests": 0,
                "total_tests": len(tests),
                "error": str(exc)[:300],
            }

    total_tests = int(eval_result.get("total_tests") or len(tests) or 0)
    passed_tests = int(eval_result.get("passed_tests") or 0)
    pass_frac = passed_tests / max(total_tests, 1)
    interface_ok = bool(eval_result.get("interface_ok"))
    passed = bool(eval_result.get("passed"))
    status = str(eval_result.get("status") or "eval_error")

    # A fully correct solution gets 1.0. Any imperfect solution is capped at
    # 0.8 and scaled by its test pass fraction. For example, 70/80 -> 0.7.
    score = 1.0 if total_tests > 0 and passed else 0.8 * pass_frac

    return {
        "score": score,
        "acc": 1.0 if passed else 0.0,
        "format_ok": float(format_ok),
        "parse_ok": float(parse_ok),
        "interface_ok": float(interface_ok),
        "pass_frac": float(pass_frac),
        "passed": 1.0 if passed else 0.0,
        "passed_tests": float(passed_tests),
        "total_tests": float(total_tests),
        "wrong_answers": float(eval_result.get("wrong_answers") or 0),
        "runtime_errors": float(eval_result.get("runtime_errors") or 0),
        "timeouts": float(eval_result.get("timeouts") or 0),
        "status_code": float(STATUS_CODE.get(status, STATUS_CODE["eval_error"])),
    }
