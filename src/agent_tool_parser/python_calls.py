# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

"""Extractor for Python function call expressions in markdown and raw agent outputs.

Supports multiline argument blocks, triple-quoted strings (\"\"\" / '''),
domestic LLM dialects (GigaChat, YandexGPT), and Russian semantic action prefixes.
"""

from __future__ import annotations

import ast
import re
import warnings
from typing import Any

from agent_tool_parser.cleaners import safe_json_loads
from agent_tool_parser.models import ToolCall

_CODEBLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)\s*```", re.DOTALL)

_PYTHON_BUILTINS = {
    "print",
    "len",
    "range",
    "dict",
    "list",
    "str",
    "int",
    "float",
    "bool",
    "set",
    "tuple",
    "isinstance",
    "type",
    "enumerate",
    "zip",
    "sum",
    "min",
    "max",
    "open",
    "help",
    "id",
    "input",
    "eval",
    "exec",
    "compile",
}

_PYTHON_KEYWORDS = {
    "def",
    "class",
    "return",
    "yield",
    "raise",
    "if",
    "elif",
    "else",
    "while",
    "for",
    "try",
    "except",
    "with",
    "async",
    "lambda",
}

_PREFIXES = (
    "вызов функции:",
    "вызов инструмента:",
    "действие:",
    "вызов:",
    "инструмент:",
    "функция:",
    "action:",
    "call:",
    "tool:",
    "function:",
)


def _normalize_raw_func_name(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("call:"):
        raw = raw[5:].strip()
    if "." in raw:
        parts = raw.split(".")
        if len(parts) >= 2 and parts[1] in ("call", "run", "execute", "invoke"):
            return parts[0]
        elif len(parts) >= 2 and parts[0] in ("tool", "tools", "action", "functions"):
            return parts[1]
        else:
            return parts[-1]
    return raw


def _find_matching_close_paren(text: str, open_idx: int) -> int | None:
    depth = 0
    in_triple_double = False
    in_triple_single = False
    in_double = False
    in_single = False
    escape = False

    n = len(text)
    j = open_idx

    while j < n:
        c = text[j]

        if escape:
            escape = False
            j += 1
            continue

        if c == "\\" and (in_triple_double or in_triple_single or in_double or in_single):
            escape = True
            j += 1
            continue

        if not in_single and not in_double:
            if not in_triple_single and j + 2 < n and text[j : j + 3] == '"""':
                in_triple_double = not in_triple_double
                j += 3
                continue
            if not in_triple_double and j + 2 < n and text[j : j + 3] == "'''":
                in_triple_single = not in_triple_single
                j += 3
                continue

        if not in_triple_double and not in_triple_single:
            if c == '"' and not in_single:
                in_double = not in_double
                j += 1
                continue
            if c == "'" and not in_double:
                in_single = not in_single
                j += 1
                continue

        if not in_triple_double and not in_triple_single and not in_double and not in_single:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return j

        j += 1

    return None


def _split_python_args(raw_args: str) -> list[str]:
    tokens: list[str] = []
    cur: list[str] = []
    chars = list(raw_args)
    n = len(chars)
    i = 0

    in_triple_double = False
    in_triple_single = False
    in_double = False
    in_single = False
    escape = False
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0

    while i < n:
        c = chars[i]

        if escape:
            cur.append(c)
            escape = False
            i += 1
            continue

        if c == "\\" and (in_triple_double or in_triple_single or in_double or in_single):
            cur.append(c)
            escape = True
            i += 1
            continue

        if not in_single and not in_double:
            if (
                not in_triple_single
                and i + 2 < n
                and chars[i] == '"'
                and chars[i + 1] == '"'
                and chars[i + 2] == '"'
            ):
                in_triple_double = not in_triple_double
                cur.extend(['"', '"', '"'])
                i += 3
                continue
            if (
                not in_triple_double
                and i + 2 < n
                and chars[i] == "'"
                and chars[i + 1] == "'"
                and chars[i + 2] == "'"
            ):
                in_triple_single = not in_triple_single
                cur.extend(["'", "'", "'"])
                i += 3
                continue

        if not in_triple_double and not in_triple_single:
            if c == '"' and not in_single:
                in_double = not in_double
                cur.append(c)
                i += 1
                continue
            if c == "'" and not in_double:
                in_single = not in_single
                cur.append(c)
                i += 1
                continue

        if not in_triple_double and not in_triple_single and not in_double and not in_single:
            if c == "(":
                paren_depth += 1
            elif c == ")":
                if paren_depth > 0:
                    paren_depth -= 1
            elif c == "[":
                bracket_depth += 1
            elif c == "]":
                if bracket_depth > 0:
                    bracket_depth -= 1
            elif c == "{":
                brace_depth += 1
            elif c == "}":
                if brace_depth > 0:
                    brace_depth -= 1
            elif c == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                tok = "".join(cur).strip()
                if tok:
                    tokens.append(tok)
                cur = []
                i += 1
                continue

        cur.append(c)
        i += 1

    tok = "".join(cur).strip()
    if tok:
        tokens.append(tok)

    return tokens


def _parse_python_arg_value(val_str: str) -> Any:
    trimmed = val_str.strip()
    if not trimmed:
        return ""

    # Try standard ast.literal_eval first
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.literal_eval(trimmed)
    except Exception:
        pass

    # Triple quoted strings
    if (trimmed.startswith('"""') and trimmed.endswith('"""') and len(trimmed) >= 6) or (
        trimmed.startswith("'''") and trimmed.endswith("'''") and len(trimmed) >= 6
    ):
        inner = trimmed[3:-3]
        return bytes(inner, "utf-8").decode("unicode_escape", errors="replace")

    # Double/single quoted strings
    if (trimmed.startswith('"') and trimmed.endswith('"') and len(trimmed) >= 2) or (
        trimmed.startswith("'") and trimmed.endswith("'") and len(trimmed) >= 2
    ):
        inner = trimmed[1:-1]
        try:
            return bytes(inner, "utf-8").decode("unicode_escape", errors="replace")
        except Exception:
            return inner

    if (trimmed.startswith("{") and trimmed.endswith("}")) or (
        trimmed.startswith("[") and trimmed.endswith("]")
    ):
        if (j := safe_json_loads(trimmed)) is not None:
            return j

    return trimmed


def _parse_python_kwargs(raw_args: str) -> dict[str, Any]:
    # Try ast.parse on dummy function call first
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(f"func({raw_args})")
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Call)
        ):
            call_node = tree.body[0].value
            args_map: dict[str, Any] = {}
            for kw in call_node.keywords:
                if kw.arg:
                    try:
                        args_map[kw.arg] = ast.literal_eval(kw.value)
                    except Exception:
                        try:
                            args_map[kw.arg] = ast.unparse(kw.value)
                        except Exception:
                            args_map[kw.arg] = ""
            for idx, arg_val in enumerate(call_node.args):
                try:
                    args_map[f"arg{idx}"] = ast.literal_eval(arg_val)
                except Exception:
                    try:
                        args_map[f"arg{idx}"] = ast.unparse(arg_val)
                    except Exception:
                        args_map[f"arg{idx}"] = ""
            return args_map
    except Exception:
        pass

    # Fallback tokenizer
    tokens = _split_python_args(raw_args)
    args_map = {}
    pos_idx = 0
    for tok in tokens:
        if not tok:
            continue
        # Find unquoted =
        eq_idx = -1
        in_str = False
        str_q = ""
        esc = False
        depth = 0
        for i, ch in enumerate(tok):
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch in ('"', "'") and not esc:
                if not in_str:
                    in_str = True
                    str_q = ch
                elif str_q == ch:
                    in_str = False
            elif not in_str:
                if ch in ("(", "[", "{"):
                    depth += 1
                elif ch in (")", "]", "}"):
                    if depth > 0:
                        depth -= 1
                elif ch == "=" and depth == 0:
                    eq_idx = i
                    break

        if eq_idx != -1:
            key = tok[:eq_idx].strip()
            val = tok[eq_idx + 1 :].strip()
            args_map[key] = _parse_python_arg_value(val)
        else:
            args_map[f"arg{pos_idx}"] = _parse_python_arg_value(tok)
            pos_idx += 1

    return args_map


def _scan_calls_in_text(
    text: str,
    allowed_tools: set[str] | None = None,
    tool_aliases: dict[str, str] | None = None,
) -> list[ToolCall]:
    calls: list[ToolCall] = []
    n = len(text)
    i = 0

    while i < n:
        if text[i] == "(":
            open_paren = i
            p = open_paren
            while p > 0 and text[p - 1].isspace():
                p -= 1
            name_end = p
            while p > 0 and (text[p - 1].isalnum() or text[p - 1] in ("_", ".")):
                p -= 1
            name_start = p

            if name_end > name_start:
                func_ident = text[name_start:name_end]
                prefix_start = name_start
                while prefix_start > 0 and text[prefix_start - 1] not in ("\n", "\r"):
                    prefix_start -= 1
                before_name = text[prefix_start:name_start]
                before_trimmed = before_name.strip()

                is_prefix_valid = not before_trimmed or before_trimmed.endswith("`")
                before_lower = before_trimmed.lower()
                for pref in _PREFIXES:
                    if before_lower.endswith(pref):
                        is_prefix_valid = True
                        break

                if "=" in before_trimmed and not is_prefix_valid:
                    i += 1
                    continue

                if is_prefix_valid:
                    func_name = _normalize_raw_func_name(func_ident)
                    is_builtin = func_name in _PYTHON_BUILTINS
                    is_keyword = func_name in _PYTHON_KEYWORDS

                    if not is_keyword and (allowed_tools is not None or not is_builtin):
                        close_paren = _find_matching_close_paren(text, open_paren)
                        if close_paren is not None:
                            raw_args_str = text[open_paren + 1 : close_paren]
                            raw_source_str = text[prefix_start : close_paren + 1].strip()

                            parsed_args = _parse_python_kwargs(raw_args_str)
                            norm_name = func_name
                            if tool_aliases and func_name.lower() in tool_aliases:
                                norm_name = tool_aliases[func_name.lower()]

                            is_allowed = (
                                allowed_tools is None
                                or norm_name in allowed_tools
                                or func_name in allowed_tools
                            )

                            if is_allowed:
                                calls.append(
                                    ToolCall(
                                        name=norm_name,
                                        args=parsed_args,
                                        raw_source=raw_source_str,
                                    )
                                )
                                i = close_paren + 1
                                continue
        i += 1

    return calls


def extract_python_function_calls(
    text: str,
    allowed_tools: set[str] | None = None,
    tool_aliases: dict[str, str] | None = None,
) -> list[ToolCall]:
    """Scans text for Python function call expressions in codeblocks or bare text."""
    calls: list[ToolCall] = []

    for m in _CODEBLOCK_RE.finditer(text):
        block = m.group(1)
        sub = _scan_calls_in_text(block, allowed_tools, tool_aliases)
        calls.extend(sub)

    if calls:
        return calls

    return _scan_calls_in_text(text, allowed_tools, tool_aliases)
