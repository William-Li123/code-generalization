#!/usr/bin/env python3
"""Dependency-free helpers for the frozen Stage-02 prompt contract."""

from __future__ import annotations

from typing import Any


def apply_native_template(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    """Render a native template with thinking explicitly disabled."""

    return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
