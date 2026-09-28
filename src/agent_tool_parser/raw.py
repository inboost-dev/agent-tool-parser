# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent_tool_parser.cleaners import (
    clean_param_val,
    extract_json_objects,
    safe_json_loads,
    strip_thinking,
)
from agent_tool_parser.models import ToolCall

_ATTR_NAME_RE = re.compile(
    r"\b(?:name|tool|tool_name|function)\s*=\s*['\"]?([a-zA-Z0-9_.:-]+)['\"]?",
    re.IGNORECASE,
)

_INVOKE_RE = re.compile(
    r"<[｜|]*(?:dsml[｜|]*)?(?P<tag>tool_invoke|invoke|tool_call|call|tool|invocation|function_call|function|action|tool_use|ant_tool_use|function_use)(?::(?P<colon_tool>[\w-]+))?\b(?P<attrs>[^>]*)>(?P<body>.*?)(?:</[｜|]*(?:dsml[｜|]*)?(?:tool_invoke|invoke|tool_call|call|tool|invocation|function_call|function|action|tool_use|ant_tool_use|function_use)(?::[\w-]+)?\s*>|<[｜|]*(?:dsml[｜|]*)?(?:tool_invoke|invoke|tool_call|call|tool|invocation|function_call|function|action|tool_use|ant_tool_use|function_use)\b|</[｜|]*(?:dsml[｜|]*)?(?:tool_calls|function_calls|calls|tools)\s*>|$)",
    re.IGNORECASE | re.DOTALL,
)

_NATIVE_DEEPSEEK_RE = re.compile(
    r"<[｜|]*tool call begin[｜|]*>(?:function=)?(?P<name>[\w.:-]+)<[｜|]*tool sep[｜|]*>(?P<args>.*?)<[｜|]*tool call end[｜|]*>",
    re.DOTALL,
)

_QWEN_ACTION_RE = re.compile(
    r"<\|action_start\|><\|action_name\|>(?P<name>[\w.:-]+)<\|action_args\|>(?P<args>.*?)<\|action_end\|>",
    re.IGNORECASE | re.DOTALL,
)

_REACT_ACTION_RE = re.compile(
    r"Action:\s*(?P<name>[\w.:-]+)\s*\nAction Input:\s*(?P<args>[^\n]+(?:\n[^\n]+)*)",
    re.IGNORECASE | re.DOTALL,
)

_MISTRAL_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[\s*\{.*?\}\s*\])", re.IGNORECASE | re.DOTALL)

_PARAM_START_RE = re.compile(
    r"<[｜|]*(?:dsml[｜|]*)?(?:parameter|param|arg|argument)\b[^>]*?\bname\s*=\s*['\"]?(?P<pname>[\w-]+)['\"]?[^>]*>",
    re.IGNORECASE | re.DOTALL,
)

_PARAM_END_RE = re.compile(
    r"</[｜|]*(?:dsml[｜|]*)?(?:parameter|param|arg|argument)\s*>",
    re.IGNORECASE | re.DOTALL,
)

_PARAM_END_FALLBACK_RE = re.compile(
    r"</[｜|]+(?:dsml[｜|]*)?[\w:-]+\s*>|</dsml:[\w:-]+\s*>|</(?:tool_name|function_name|tool|invoke|parameter|param|arg|argument)\s*>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class RawToolCall:
    """Zero-copy capable raw representation of an extracted tool call.

    Holds arguments as unparsed raw strings, allowing multi-GB/s structural routing,
    inspection, and authorization before full deserialization.
    """

    name: str
    raw_args: str
    raw_source: str
    call_id: str | None = None

    def parse_args(self) -> dict[str, Any] | list[Any] | str:
        """Parse raw arguments into structured dictionary or value on demand."""
        trimmed = self.raw_args.strip()

        # 1. Parse XML/DSML parameter tags
        if _PARAM_START_RE.search(trimmed):
            args_map: dict[str, Any] = {}
            cdata_spans: list[tuple[int, int]] = []
            cur = 0
            while True:
                start = trimmed.find("<![CDATA[", cur)
                if start == -1:
                    break
                end = trimmed.find("]]>", start)
                if end != -1:
                    cdata_spans.append((start, end + 3))
                    cur = end + 3
                else:
                    cdata_spans.append((start, len(trimmed)))
                    break

            matches = list(_PARAM_START_RE.finditer(trimmed))
            valid_starts: list[tuple[int, int, str]] = []
            for m in matches:
                start = m.start()
                if any(cs <= start < ce for cs, ce in cdata_spans):
                    continue
                pname = m.group("pname").strip().lower()
                valid_starts.append((start, m.end(), pname))

            for i, (_, val_start, pname) in enumerate(valid_starts):
                upper_bound = valid_starts[i + 1][0] if i + 1 < len(valid_starts) else len(trimmed)
                slice_str = trimmed[val_start:upper_bound]

                cdata_start = slice_str.find("<![CDATA[")
                if cdata_start != -1:
                    cdata_end = slice_str.find("]]>", cdata_start)
                    search_from = (
                        (cdata_start + cdata_end + 3) if cdata_end != -1 else len(slice_str)
                    )
                else:
                    search_from = 0

                search_slice = slice_str[search_from:]
                end_m = _PARAM_END_RE.search(search_slice) or _PARAM_END_FALLBACK_RE.search(
                    search_slice
                )
                if end_m:
                    val = slice_str[: search_from + end_m.start()]
                else:
                    val = slice_str

                is_code = any(k in pname for k in ("code", "content", "script"))
                clean_val = clean_param_val(val.strip(), is_code_param=is_code)
                parsed_val = safe_json_loads(clean_val, default=clean_val)
                args_map[pname] = parsed_val

            return args_map

        # 2. Try JSON deserialization
        if trimmed.startswith("{") or trimmed.startswith("["):
            loaded = safe_json_loads(trimmed)
            if isinstance(loaded, (dict, list)):
                return loaded
            from agent_tool_parser.models import ToolError

            raise ToolError(f"Failed to parse JSON arguments for tool '{self.name}': {trimmed}")

        # 3. String command fallback
        key = "command" if any(k in self.name for k in ("bash", "sh", "exec")) else "input"
        return {key: trimmed}

    def to_tool_call(self) -> ToolCall:
        """Convert into a full ToolCall instance."""
        args = self.parse_args()
        if not isinstance(args, dict):
            args = {"input": args}
        return ToolCall(name=self.name, args=args, raw_source=self.raw_source)


def extract_raw_tool_calls(text: str) -> list[RawToolCall]:
    """Extract raw tool calls from text without full JSON parsing overhead."""
    clean = strip_thinking(text)
    results: list[RawToolCall] = []

    # 1. DSML and XML invocation blocks
    for m in _INVOKE_RE.finditer(clean):
        full = m.group(0)
        colon_tool = m.group("colon_tool")
        attrs = m.group("attrs") or ""
        body = m.group("body") or ""

        name: str | None = None
        if colon_tool and colon_tool.strip():
            name = colon_tool.strip()
        else:
            attr_m = _ATTR_NAME_RE.search(attrs)
            if attr_m:
                name = attr_m.group(1).strip()

        if name:
            results.append(RawToolCall(name=name, raw_args=body.strip(), raw_source=full))

    if results:
        return results

    # 2. Native DeepSeek tokens
    for m in _NATIVE_DEEPSEEK_RE.finditer(clean):
        full = m.group(0)
        name = m.group("name").strip()
        args = m.group("args").strip()
        if name:
            results.append(RawToolCall(name=name, raw_args=args, raw_source=full))

    if results:
        return results

    # 3. Qwen action tokens
    for m in _QWEN_ACTION_RE.finditer(clean):
        full = m.group(0)
        name = m.group("name").strip()
        args = m.group("args").strip()
        if name:
            results.append(RawToolCall(name=name, raw_args=args, raw_source=full))

    if results:
        return results

    # 4. Mistral [TOOL_CALLS]
    mistral_m = _MISTRAL_RE.search(clean)
    if mistral_m:
        full = mistral_m.group(0)
        arr_str = mistral_m.group(1)
        loaded = safe_json_loads(arr_str)
        if isinstance(loaded, list):
            for item in loaded:
                if isinstance(item, dict) and "name" in item:
                    raw_args = item.get("arguments", "")
                    if not isinstance(raw_args, str):
                        import json

                        raw_args = json.dumps(raw_args)
                    results.append(
                        RawToolCall(name=item["name"], raw_args=raw_args, raw_source=full)
                    )

    if results:
        return results

    # 5. ReAct pattern
    react_m = _REACT_ACTION_RE.search(clean)
    if react_m:
        full = react_m.group(0)
        name = react_m.group("name").strip()
        args = react_m.group("args").strip()
        if name:
            results.append(RawToolCall(name=name, raw_args=args, raw_source=full))

    if results:
        return results

    # 6. JSON objects in Markdown or raw text
    json_objs = extract_json_objects(clean)
    for raw_json in json_objs:
        loaded = safe_json_loads(raw_json)
        if isinstance(loaded, dict):
            # Check standard OpenAI tool_calls structure
            if "tool_calls" in loaded and isinstance(loaded["tool_calls"], list):
                for c in loaded["tool_calls"]:
                    if isinstance(c, dict) and "function" in c and isinstance(c["function"], dict):
                        func = c["function"]
                        name = func.get("name")
                        if name:
                            call_id = c.get("id")
                            raw_args = func.get("arguments", "")
                            if not isinstance(raw_args, str):
                                import json

                                raw_args = json.dumps(raw_args)
                            results.append(
                                RawToolCall(
                                    name=name,
                                    raw_args=raw_args,
                                    raw_source=raw_json,
                                    call_id=call_id,
                                )
                            )

            if results:
                return results

            # Direct object with tool name key
            name = (
                loaded.get("name")
                or loaded.get("tool")
                or loaded.get("action")
                or loaded.get("function")
            )
            if name and isinstance(name, str):
                raw_args_val = (
                    loaded.get("arguments") or loaded.get("parameters") or loaded.get("args")
                )
                if raw_args_val is not None:
                    import json

                    raw_args = (
                        json.dumps(raw_args_val)
                        if not isinstance(raw_args_val, str)
                        else raw_args_val
                    )
                else:
                    raw_args = raw_json
                results.append(RawToolCall(name=name, raw_args=raw_args, raw_source=raw_json))

    return results


def try_extract_raw_tool_call(text: str) -> RawToolCall | None:
    """Try to extract a single raw tool call from text."""
    calls = extract_raw_tool_calls(text)
    return calls[0] if calls else None
