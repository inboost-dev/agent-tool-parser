# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from typing import Any


def canonical_sort_keys(data: Any) -> Any:
    """Recursively sort dictionary keys for deterministic serialization.

    Ensures that identical payloads always yield identical JSON byte representations,
    maximizing KV-cache prefix hits across inference engines (vLLM, SGLang, Ollama, DeepSeek).
    """
    if isinstance(data, dict):
        return {k: canonical_sort_keys(v) for k, v in sorted(data.items())}
    if isinstance(data, list):
        return [canonical_sort_keys(item) for item in data]
    return data


def canonical_json_dumps(data: Any) -> str:
    """Serialize data to a deterministic, canonical JSON string with sorted keys and compact separators."""
    sorted_data = canonical_sort_keys(data)
    return json.dumps(sorted_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonicalize_arguments_string(args_str: str) -> str:
    """Parse an argument string, sort its keys recursively, and re-serialize.

    If the string is not valid JSON, returns it trimmed without modifications.
    """
    if not args_str or not isinstance(args_str, str):
        return args_str
    stripped = args_str.strip()
    if not (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    ):
        return stripped
    try:
        parsed = json.loads(stripped)
        return canonical_json_dumps(parsed)
    except Exception:
        return stripped


def canonicalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize a list of OpenAI-style tool calls for KV-cache invariance.

    1. Recursively sorts keys inside `function.arguments`.
    2. Stably sorts multiple tool calls by function name and ID to guarantee order invariance.
    """
    if not tool_calls:
        return tool_calls

    aligned_calls = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            aligned_calls.append(tc)
            continue
        c = dict(tc)
        fn = c.get("function")
        if isinstance(fn, dict) and "arguments" in fn:
            args = fn["arguments"]
            if isinstance(args, str):
                c["function"] = dict(fn)
                c["function"]["arguments"] = canonicalize_arguments_string(args)
            elif isinstance(args, dict):
                c["function"] = dict(fn)
                c["function"]["arguments"] = canonical_json_dumps(args)
        aligned_calls.append(c)

    # Stable sort by function name, then by call ID
    aligned_calls.sort(
        key=lambda x: (
            x.get("function", {}).get("name", "") if isinstance(x.get("function"), dict) else "",
            str(x.get("id", "")),
        )
    )
    return aligned_calls
