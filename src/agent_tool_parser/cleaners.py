# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

from __future__ import annotations

import html
import json
import re
from typing import Any

_CODE_PARAM_KEYS_DEFAULT: set[str] = {
    "old_str",
    "new_str",
    "content",
    "old",
    "new",
    "code",
    "text",
    "replacement",
    "patch",
    "diff",
}

_THINK_OPEN_RE = re.compile(r"<[｜|]?(?:think|thought|reasoning)[｜|]?>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</[｜|]?(?:think|thought|reasoning)[｜|]?>", re.IGNORECASE)

_TOOL_START_CANDIDATES = (
    "<tool_call",
    "<tool_calls",
    "<tools",
    "<tool",
    "<invoke",
    "<dsml",
    "<call",
    "<tool_use",
    "<ant_tool_use",
    "<action_start",
    "<action_name",
    "<python_tag",
    "```",
    "action:",
    "✿function✿",
)

_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)(?:\]\]>|$)", re.DOTALL)
_SINGLE_QUOTE_KEYS_RE = re.compile(r"([{,]\s*)'([^']+)'(\s*:)")
_SINGLE_QUOTE_VALS_RE = re.compile(r"(:\s*)'([^']*)'(\s*[,}])")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_PY_TRUE_RE = re.compile(r"\bTrue\b")
_PY_FALSE_RE = re.compile(r"\bFalse\b")
_PY_NONE_RE = re.compile(r"\bNone\b")

_BRACE_COLON_RE = re.compile(r"^{\s*\"(?::|\s*:)\s*")
_PREFIX_COLON_RE = re.compile(r"^\s*\"(?::|\s*:)\s*")
_RAW_COLON_RE = re.compile(r"^\s*:\s*")


def normalize_truncated_json_prefix(s: str) -> str:
    """Normalizes truncated JSON prefixes (e.g. from LLMs dropping `{"` or `{"tool`)."""
    trimmed = s.lstrip()
    offset = len(s) - len(trimmed)
    prefix = s[:offset]
    if trimmed.startswith(('tool":', 'name":', 'action":')):
        return prefix + '{"' + trimmed
    elif trimmed.startswith(('"tool":', '"name":', '"action":')):
        return prefix + "{" + trimmed
    elif m := _BRACE_COLON_RE.match(trimmed):
        return prefix + '{"tool": ' + trimmed[m.end() :]
    elif m := _PREFIX_COLON_RE.match(trimmed):
        return prefix + '{"tool": ' + trimmed[m.end() :]
    elif m := _RAW_COLON_RE.match(trimmed):
        return prefix + '{"tool": ' + trimmed[m.end() :]
    return s


def strip_thinking(text: str) -> str:
    """Strips internal reasoning / thinking tags (<think>...</think>)."""
    if "<" not in text:
        return text

    open_m = _THINK_OPEN_RE.search(text)
    if open_m:
        after_open = text[open_m.end() :]
        # 1. Look for explicit closing tag
        close_m = _THINK_CLOSE_RE.search(after_open)
        if close_m:
            before = text[: open_m.start()]
            after = after_open[close_m.end() :]
            result = before + after
            if _THINK_OPEN_RE.search(result):
                return strip_thinking(result)
            return result.strip()

        # 2. No closing tag: find first tool candidate
        lower = after_open.lower()
        min_pos = None
        for cand in _TOOL_START_CANDIDATES:
            pos = lower.find(cand)
            if pos != -1:
                if min_pos is None or pos < min_pos:
                    min_pos = pos
        if min_pos is not None:
            before = text[: open_m.start()]
            after = after_open[min_pos:]
            return (before + after).strip()

        before = text[: open_m.start()].strip()
        if before:
            return before

    return text


def clean_param_val(val: str, is_code_param: bool = False) -> str:
    """Cleans an extracted XML/DSML parameter value.

    1. Unwraps <![CDATA[...]]> blocks if present (preserving raw code).
    2. Unescapes HTML/XML entities (&amp;, &lt;, &gt;, &quot;) only if not in CDATA.
    3. Trims whitespace appropriately: preserves indentation for code/diffs,
       stripping only leading/trailing bounding newlines.
    """
    has_cdata = False
    if "<![CDATA[" in val:
        m_cdata = _CDATA_RE.search(val)
        if m_cdata:
            val = m_cdata.group(1)
            has_cdata = True

    if not has_cdata and "&" in val:
        val = html.unescape(val)

    if is_code_param:
        val = val.removeprefix("\r\n").removeprefix("\n").removesuffix("\r\n").removesuffix("\n")
    else:
        val = val.strip()

    return val


def clean_json_str(s: str) -> str:
    """Applies heuristic repairs to malformed JSON strings.

    - Replaces single-quoted keys and string values with double quotes.
    - Strips trailing commas before closing braces/brackets.
    - Normalizes Python literals (True, False, None) to JSON (true, false, null).
    - Normalizes truncated JSON prefixes (e.g. tool": or ":).
    """
    s = normalize_truncated_json_prefix(s)
    if "'" in s:
        if '"' not in s:
            s = s.replace("'", '"')
        else:
            # Single-quoted keys: {'key': ...} -> {"key": ...}
            s = _SINGLE_QUOTE_KEYS_RE.sub(r'\g<1>"\g<2>"\g<3>', s)

            # Single-quoted values: : 'val' -> : "val" (escaping any inner double quotes)
            def _replace_sq_val(m: re.Match[str]) -> str:
                prefix, val, suffix = m.group(1), m.group(2), m.group(3)
                val_escaped = val.replace('\\"', '"').replace('"', '\\"')
                return f'{prefix}"{val_escaped}"{suffix}'

            s = _SINGLE_QUOTE_VALS_RE.sub(_replace_sq_val, s)

    # Fix trailing commas: {"a": 1,} -> {"a": 1}
    if "," in s:
        s = _TRAILING_COMMA_RE.sub(r"\g<1>", s)

    # Replace Python literals
    if "True" in s:
        s = _PY_TRUE_RE.sub("true", s)
    if "False" in s:
        s = _PY_FALSE_RE.sub("false", s)
    if "None" in s:
        s = _PY_NONE_RE.sub("null", s)
    return s


try:
    import orjson  # type: ignore[import-not-found,import-untyped]

    def _fast_json_loads(s: str) -> Any:
        return orjson.loads(s)
except ImportError:
    import json

    def _fast_json_loads(s: str) -> Any:
        return json.loads(s, strict=False)


try:
    import json_repair  # type: ignore[import-not-found,import-untyped]

    def _repair_json_ext(s: str) -> Any:
        return json_repair.repair_json(s, return_objects=True)
except ImportError:
    _repair_json_ext = None  # type: ignore[assignment]


def safe_json_loads(s: str, default: Any = None) -> Any:
    """Safely parses JSON with fallback to heuristic repair, returning default on failure.

    Args:
        s: Raw JSON string or malformed candidate.
        default: Fallback value if parsing fails completely.

    Returns:
        Parsed JSON object or default value.
    """
    if not s or not isinstance(s, str):
        return default
    try:
        return _fast_json_loads(s)
    except Exception:
        pass

    try:
        return _fast_json_loads(clean_json_str(s))
    except Exception:
        pass

    if _repair_json_ext is not None:
        try:
            res = _repair_json_ext(s)
            if isinstance(res, (dict, list)):
                return res
        except Exception:
            pass

    return default


def extract_json_objects(text: str) -> list[str]:
    """Extracts all balanced JSON object candidates from arbitrary text using bracket-depth tracking.

    Handles double and single quoted strings, escape characters, and auto-completes
    truncated objects (including unclosed arrays and nested objects) if the LLM stream
    was cut off mid-generation.
    """
    text = normalize_truncated_json_prefix(text)
    if "{" not in text:
        return []

    results: list[str] = []
    stack: list[str] = []
    start = -1
    in_string = False
    quote_char: str | None = None
    escape = False

    for i, char in enumerate(text):
        if char in ('"', "'") and not escape:
            if not in_string:
                in_string = True
                quote_char = char
            elif quote_char == char:
                in_string = False
                quote_char = None
        elif char == "\\" and in_string:
            escape = not escape
            continue
        elif not in_string:
            if char == "{":
                if not stack:
                    start = i
                stack.append("{")
            elif char == "[":
                if stack:
                    stack.append("[")
            elif char == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
                    if not stack and start != -1:
                        results.append(text[start : i + 1])
                        start = -1
                elif stack and "{" in stack:
                    while stack and stack[-1] != "{":
                        stack.pop()
                    if stack:
                        stack.pop()
                        if not stack and start != -1:
                            results.append(text[start : i + 1])
                            start = -1
            elif char == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
        if escape:
            escape = False

    # Auto-repair truncated tail if JSON generation was cut off mid-stream
    if stack and start != -1:
        tail = text[start:]
        if in_string and quote_char:
            if escape or tail.endswith("\\"):
                tail = tail.rstrip("\\")
            tail += quote_char
        trimmed_tail = tail.rstrip()
        if trimmed_tail.endswith(":"):
            tail = trimmed_tail + '""'
        elif trimmed_tail.endswith(","):
            tail = trimmed_tail[:-1]
        for b in reversed(stack):
            if b == "[":
                tail += "]"
            elif b == "{":
                tail += "}"
        results.append(tail)

    return results
