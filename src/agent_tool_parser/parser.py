# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Collection
from typing import Any

from agent_tool_parser.cleaners import (
    _CODE_PARAM_KEYS_DEFAULT,
    clean_param_val,
    extract_json_objects,
    safe_json_loads,
    strip_thinking,
)
from agent_tool_parser.models import ToolCall, ToolError

_DEFAULT_ALIASES: dict[str, str] = {
    "readfile": "read_file",
    "writefile": "write_file",
    "strreplace": "str_replace",
    "tool_search": "search",
    "tool_bash": "bash",
    "tool_read_file": "read_file",
    "tool_write_file": "write_file",
    "tool_str_replace": "str_replace",
    "grep": "search",
    "grep_search": "search",
    "view": "read_file",
    "view_file": "read_file",
    "edit": "str_replace",
    "edit_file": "str_replace",
    "sh": "bash",
    "shell": "bash",
    "terminal": "bash",
}

_DEFAULT_PARAM_ALIASES: dict[str, str] = {
    "cmd": "command",
    "file": "path",
    "filepath": "path",
    "file_path": "path",
    "target_file": "path",
    "filename": "path",
    "query": "pattern",
    "old": "old_str",
    "new": "new_str",
}

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_ARRAY_BLOCK_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)

_INVOKE_TAGS = (
    r"tool_invoke|invoke|tool_call|call|tool|invocation|"
    r"function_call|function|action|tool_use|ant_tool_use|function_use"
)

_INVOKE_RE = re.compile(
    r"""<[｜|]*(?:dsml[｜|]*)?(?P<tag>""" + _INVOKE_TAGS + r""")"""
    r"""(?::(?P<colon_tool>[\w-]+))?\b(?P<attrs>[^>]*)>(?P<body>.*?)"""
    r"""(?:</[｜|]*(?:dsml[｜|]*)?(?P=tag)(?::[\w-]+)?\s*>|"""
    r"""(?=<[｜|]*(?:dsml[｜|]*)?(?:""" + _INVOKE_TAGS + r""")\b)|"""
    r"""(?=</[｜|]*(?:dsml[｜|]*)?(?:tool_calls|function_calls|calls|tools)\s*>)|$)""",
    re.DOTALL | re.IGNORECASE,
)

_PARAM_RE = re.compile(
    r"""<[｜|]*(?:dsml[｜|]*)?(?P<ptag>parameter|param|arg|argument)\b[^>]*?\bname\s*=\s*['"]?(?P<pname>[\w-]+)['"]?[^>]*>(?P<pval>.*?)"""
    r"""(?:</[｜|]*(?:dsml[｜|]*)?(?P=ptag)\s*>|"""
    r"""(?=<[｜|]*(?:dsml[｜|]*)?(?:parameter|param|arg|argument)\b)|"""
    r"""(?=</?[｜|]*(?:dsml[｜|]*)?(?:"""
    + _INVOKE_TAGS
    + r"""|tool_calls|function_calls|calls|tools)\b)|$)""",
    re.DOTALL | re.IGNORECASE,
)

_PARAM_START_RE = re.compile(
    r"""<[｜|]*(?:dsml[｜|]*)?(?:parameter|param|arg|argument)\b[^>]*?\bname\s*=\s*['"]?(?P<pname>[\w-]+)['"]?[^>]*>""",
    re.IGNORECASE,
)

_PARAM_END_RE = re.compile(
    r"""</[｜|]*(?:dsml[｜|]*)?(?:parameter|param|arg|argument)\s*>""",
    re.IGNORECASE,
)

_INVOKE_END_RE = re.compile(
    r"""</[｜|]*(?:dsml[｜|]*)?(?:"""
    + _INVOKE_TAGS
    + r""")(?:(?::[\w-]+)?\s*>)|</[｜|]*(?:dsml[｜|]*)?(?:tool_calls|function_calls|calls|tools)\s*>""",
    re.IGNORECASE,
)

_PARAM_END_FALLBACK_RE = re.compile(
    r"""</[｜|]+(?:dsml[｜|]*)?[\w:-]+\s*>|</dsml:[\w:-]+\s*>|</(?:tool_name|function_name|tool|invoke|parameter|param|arg|argument)\s*>""",
    re.IGNORECASE,
)

_HEADLESS_COLON_RE = re.compile(
    r"""(?:^|[\s`"'])(?P<delim>":|:|':)\s*["'](?P<tname>[\w-]+)["']\s*,\s*(?P<rest>.*)""",
    re.DOTALL | re.IGNORECASE,
)

_HEADLESS_KEY_RE = re.compile(
    r"""(?:^|[\s`"'])(?P<open>["'])?(?P<key>tool|name|function|action|tool_name|function_name)(?P<close>["'])?\s*:\s*(?P<rest>.*)""",
    re.DOTALL | re.IGNORECASE,
)


def _extract_xml_parameters(body: str) -> list[tuple[str, str]]:
    cdata_spans: list[tuple[int, int]] = []
    cur = 0
    while True:
        idx = body.find("<![CDATA[", cur)
        if idx == -1:
            break
        end_idx = body.find("]]>", idx)
        if end_idx != -1:
            cdata_spans.append((idx, end_idx + 3))
            cur = end_idx + 3
        else:
            cdata_spans.append((idx, len(body)))
            break

    starts: list[tuple[int, int, str]] = []
    for m in _PARAM_START_RE.finditer(body):
        s = m.start()
        if any(cs <= s < ce for cs, ce in cdata_spans):
            continue
        starts.append((s, m.end(), m.group("pname").strip().lower()))

    results: list[tuple[str, str]] = []
    for i in range(len(starts)):
        _, val_start, pname = starts[i]
        upper_bound = starts[i + 1][0] if i + 1 < len(starts) else len(body)
        slice_str = body[val_start:upper_bound]
        cdata_start = slice_str.find("<![CDATA[")
        if cdata_start != -1:
            cdata_end = slice_str.find("]]>", cdata_start)
            search_from = (cdata_end + 3) if cdata_end != -1 else len(slice_str)
        else:
            search_from = 0

        m_end = _PARAM_END_RE.search(slice_str, search_from)
        if m_end:
            val = slice_str[: m_end.start()]
        else:
            m_inv_end = _INVOKE_END_RE.search(slice_str, search_from)
            if m_inv_end:
                val = slice_str[: m_inv_end.start()]
            else:
                m_fallback = _PARAM_END_FALLBACK_RE.search(slice_str, search_from)
                if m_fallback:
                    val = slice_str[: m_fallback.start()]
                else:
                    val = slice_str
        results.append((pname, val))
    return results


_QWEN_AGENT_RE = re.compile(
    r"""(?:<[｜|]*action_start[｜|]*>\s*)?<[｜|]*action_name[｜|]*>\s*(?P<name>[\w-]+)\s*"""
    r"""<[｜|]*action_args[｜|]*>\s*(?P<args>.*?)(?:<[｜|]*action_end[｜|]*>|$)""",
    re.DOTALL | re.IGNORECASE,
)

_CHATGLM_RE = re.compile(
    r"""(?:✿|\b)FUNCTION(?:✿|\b)\s*:?\s*(?P<name>[\w-]+)\s*"""
    r"""(?:✿|\b)ARGS(?:✿|\b)\s*:?\s*(?P<args>.*)""",
    re.DOTALL | re.IGNORECASE,
)

_LLAMA_PYTHON_TAG_RE = re.compile(
    r"""<[｜|]*python_tag[｜|]*>(?P<body>.*?)(?:<[｜|]*python_tag[｜|]*>|<[｜|]*eom_id[｜|]*>|<[｜|]*eot_id[｜|]*>|$)""",
    re.DOTALL | re.IGNORECASE,
)

_MISTRAL_TOOL_CALL_RE = re.compile(
    r"""\[TOOL_CALLS?\]\s*(?P<body>\[.*?\]|\{.*?\})""",
    re.DOTALL | re.IGNORECASE,
)

_REACT_RE = re.compile(
    r"""Action:\s*(?P<name>[\w-]+)\s*\n\s*Action\s+Input:\s*(?P<args>.*?)(?=(?:\n\s*Action\s*:|\n\s*Thought\s*:|$))""",
    re.DOTALL | re.IGNORECASE,
)

_DIRECT_TAG_TNAME_RE = re.compile(
    r"<(?:tool_name|function_name|tool)>([\w-]+)</(?:tool_name|function_name|tool)>",
    re.IGNORECASE,
)
_TAG_PAIRS_RE = re.compile(r"<([\w-]+)>(.*?)</\1>", re.DOTALL)
_ATTR_TOOL_NAME_RE = re.compile(
    r"""\b(?:name|function|tool|action)\s*=\s*['"]?([\w-]+)['"]?""",
    re.IGNORECASE,
)
_CHILD_TOOL_NAME_RE = re.compile(
    r"<(?:name|tool_name|function_name|tool|function)>\s*([\w-]+)(?:</(?:name|tool_name|function_name|tool|function)>|\s|$)",
    re.IGNORECASE,
)
_ARGUMENTS_BLOCK_RE = re.compile(
    r"<(?:arguments|parameters)>(.*?)(?:</(?:arguments|parameters)>|$)",
    re.DOTALL | re.IGNORECASE,
)
_WRAP_TAG_MATCHES_RE = re.compile(
    r"<([\w-]+)>(.*?)(?:</\1>|(?=<[\w-]+>)|$)",
    re.DOTALL,
)
_PYTHON_CODEBLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)\s*```", re.DOTALL)
_FENCE_CODEBLOCK_RE = re.compile(r"```(?:[\w-]+)?\s*(.*?)\s*```", re.DOTALL)
_DEEPSEEK_SEP_RE = re.compile(r"<[｜|]*tool\s*sep[｜|]*>", re.IGNORECASE)
_DEEPSEEK_FN_RE = re.compile(
    r"""(?:function|name|tool|action)\s*[:=]\s*['"]?([\w-]+)['"]?""",
    re.IGNORECASE,
)
_DEEPSEEK_BLOCK_RE = re.compile(
    r"<[｜|]*tool\s*call\s*begin[｜|]*>(.*?)(?:<[｜|]*tool\s*call\s*end[｜|]*>|<[｜|]*tool\s*calls?\s*end[｜|]*>|$)",
    re.DOTALL | re.IGNORECASE,
)
_DEEPSEEK_TRIGGER_RE = re.compile(r"tool\s*calls?\s*begin|tool\s*sep", re.IGNORECASE)
_STANDALONE_TRIGGER_RE = re.compile(r"dsml|tool_calls|function_calls", re.IGNORECASE)


_PYTHON_BUILTINS_IGNORE = {
    "print",
    "len",
    "range",
    "str",
    "int",
    "float",
    "list",
    "dict",
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
}


class ToolParser:
    """Configurable, robust multi-format LLM tool-call parser."""

    def __init__(
        self,
        allowed_tools: Collection[str] | None = None,
        tool_aliases: dict[str, str] | None = None,
        param_aliases: dict[str, str] | None = None,
        code_param_keys: Collection[str] | None = None,
        strip_thinking: bool = True,
        allow_shell_fallback: bool = False,
    ) -> None:
        self.allowed_tools: set[str] | None = (
            set(allowed_tools) if allowed_tools is not None else None
        )
        self.tool_aliases: dict[str, str] = dict(_DEFAULT_ALIASES)
        if tool_aliases:
            self.tool_aliases.update({k.lower(): v.lower() for k, v in tool_aliases.items()})

        self.param_aliases: dict[str, str] = dict(_DEFAULT_PARAM_ALIASES)
        if param_aliases:
            self.param_aliases.update({k.lower(): v for k, v in param_aliases.items()})

        self.code_param_keys: set[str] = (
            set(code_param_keys) if code_param_keys is not None else set(_CODE_PARAM_KEYS_DEFAULT)
        )
        self.strip_thinking: bool = strip_thinking
        self.allow_shell_fallback: bool = allow_shell_fallback

    def normalize_name(self, name: str) -> str:
        """Normalizes tool name to lowercase and applies registered aliases."""
        n = (name or "").strip().lower()
        return self.tool_aliases.get(n, n)

    def normalize_args(self, name: str, args: Any) -> dict[str, Any]:
        """Unpacks nested wrappers and normalizes parameter names & types."""
        if isinstance(args, str):
            args = safe_json_loads(args, default={})

        if not isinstance(args, dict):
            return {}

        norm_args = dict(args)

        # Unpack inner wrapper dictionaries (e.g. {"arguments": {...}})
        for wrap_key in (
            "arguments",
            "parameters",
            "params",
            "args",
            "action_input",
            "tool_input",
            "input",
        ):
            if wrap_key in norm_args:
                sub = norm_args.pop(wrap_key)
                if isinstance(sub, str):
                    sub = safe_json_loads(sub, default={})
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        norm_args.setdefault(k, v)

        # Map positional argument keys if present
        if name in ("bash", "sh", "shell", "terminal") and "arg0" in norm_args:
            norm_args.setdefault("command", norm_args.pop("arg0"))
        elif name in ("search", "grep") and "arg0" in norm_args:
            norm_args.setdefault("pattern", norm_args.pop("arg0"))
        elif name in ("read_file", "view") and "arg0" in norm_args:
            norm_args.setdefault("path", norm_args.pop("arg0"))
        elif name in ("write_file",) and "arg0" in norm_args:
            norm_args.setdefault("path", norm_args.pop("arg0"))
            if "arg1" in norm_args:
                norm_args.setdefault("content", norm_args.pop("arg1"))

        # Apply parameter name aliases
        final_args: dict[str, Any] = {}
        for k, v in norm_args.items():
            key_str = str(k)
            norm_k = self.param_aliases.get(key_str.lower(), key_str)
            final_args[norm_k] = v

        # Convert numeric string parameters to int where expected
        for int_key in ("offset", "limit", "line_number", "line", "timeout"):
            if int_key in final_args:
                val = final_args[int_key]
                if isinstance(val, str) and val.strip().lstrip("-").isdigit():
                    try:
                        final_args[int_key] = int(val.strip())
                    except (ValueError, TypeError):
                        pass

        return final_args

    def _is_tool_allowed(self, name: str) -> bool:
        if not name or not name.strip():
            return False
        if self.allowed_tools is None:
            return True
        return name in self.allowed_tools

    def _extract_name_and_args_from_dict(self, data: Any) -> tuple[str | None, dict[str, Any]]:
        """Extracts tool name and arguments dictionary from diverse JSON structures
        (OpenAI, Anthropic, Command-R, standard tool dicts)."""
        if isinstance(data, list):
            if not data:
                return None, {}
            return self._extract_name_and_args_from_dict(data[0])

        if not isinstance(data, dict):
            return None, {}

        # 1. OpenAI tool_calls wrapper: {"tool_calls": [{"type": "function", "function": ...}]}
        if "tool_calls" in data and isinstance(data["tool_calls"], list) and data["tool_calls"]:
            return self._extract_name_and_args_from_dict(data["tool_calls"][0])

        # 2. OpenAI function_call wrapper: {"function_call": {"name": ..., "arguments": ...}}
        if "function_call" in data and isinstance(data["function_call"], dict):
            return self._extract_name_and_args_from_dict(data["function_call"])

        # 3. OpenAI function object: {"type": "function", "function": {"name": ..., "arguments": ...}}
        if "function" in data and isinstance(data["function"], dict):
            return self._extract_name_and_args_from_dict(data["function"])

        raw_name = None
        for k in ("name", "tool", "tool_name", "action", "function"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                raw_name = v.strip()
                break

        args_val = None
        for k in (
            "arguments",
            "parameters",
            "params",
            "args",
            "action_input",
            "tool_input",
        ):
            if k in data:
                args_val = data[k]
                break

        args: dict[str, Any] = {}
        if args_val is not None:
            if isinstance(args_val, str):
                parsed = safe_json_loads(args_val, default=None)
                if isinstance(parsed, dict):
                    args = parsed
                elif parsed is not None:
                    args = {"input": parsed}
                else:
                    args = {"input": args_val}
            elif isinstance(args_val, dict):
                args = dict(args_val)
            else:
                args = {"input": args_val}

        # Collect any remaining root arguments
        extra_keys = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "name",
                "tool",
                "tool_name",
                "action",
                "function",
                "type",
                "id",
                "arguments",
                "parameters",
                "params",
                "args",
                "action_input",
                "tool_input",
            }
        }
        for k, v in extra_keys.items():
            args.setdefault(k, v)

        return (str(raw_name).strip() if raw_name else None), args

    def _extract_calls_from_dict(self, data: Any, source: str = "") -> list[ToolCall]:
        """Extracts one or more tool calls from arbitrary JSON structures."""
        calls: list[ToolCall] = []

        if isinstance(data, list):
            for item in data:
                calls.extend(self._extract_calls_from_dict(item, source=source))
            return calls

        if not isinstance(data, dict):
            return calls

        # 1. OpenAI tool_calls wrapper: {"tool_calls": [...]}
        if "tool_calls" in data and isinstance(data["tool_calls"], list) and data["tool_calls"]:
            for item in data["tool_calls"]:
                calls.extend(self._extract_calls_from_dict(item, source=source))
            return calls

        # 2. OpenAI function_call wrapper: {"function_call": {...}}
        if "function_call" in data and isinstance(data["function_call"], dict):
            return self._extract_calls_from_dict(data["function_call"], source=source)

        # 3. Single tool call
        raw_name, args = self._extract_name_and_args_from_dict(data)
        if raw_name:
            name = self.normalize_name(raw_name)
            if self._is_tool_allowed(name):
                calls.append(
                    ToolCall(
                        name=name,
                        args=self.normalize_args(name, args),
                        raw_source=source,
                    )
                )

        return calls

    def _try_parse_deepseek_native(self, text: str) -> list[ToolCall]:
        """Parses DeepSeek V3 / R1 native format with special tokens:
        <｜tool calls begin｜><｜tool call begin｜>function=...<｜tool sep｜>{...}<｜tool call end｜>
        """
        if "tool" not in text.lower():
            return []
        if not _DEEPSEEK_TRIGGER_RE.search(text):
            return []

        calls: list[ToolCall] = []
        matches = list(_DEEPSEEK_BLOCK_RE.finditer(text))

        bodies = [m.group(1).strip() for m in matches if m.group(1).strip()]
        if not bodies and _DEEPSEEK_SEP_RE.search(text):
            bodies = [text.strip()]

        for body in bodies:
            tool_name = None
            json_str = body

            if _DEEPSEEK_SEP_RE.search(body):
                parts = _DEEPSEEK_SEP_RE.split(body, maxsplit=1)
                header = parts[0].strip()
                json_str = parts[1].strip()
                m_fn = _DEEPSEEK_FN_RE.search(header)
                if m_fn:
                    tool_name = m_fn.group(1)
                elif header and header != "type=function":
                    tool_name = header

            # Extract JSON from json_str
            m_code = _JSON_BLOCK_RE.search(json_str)
            if m_code:
                json_str = m_code.group(1)
            else:
                candidates = extract_json_objects(json_str)
                if candidates:
                    json_str = candidates[0]

            data = safe_json_loads(json_str, default={})
            if isinstance(data, dict):
                d_name, d_args = self._extract_name_and_args_from_dict(data)
                if not tool_name and d_name:
                    tool_name = d_name
                args = d_args
            else:
                args = {}

            if tool_name:
                norm_name = self.normalize_name(tool_name)
                if self._is_tool_allowed(norm_name):
                    calls.append(
                        ToolCall(
                            name=norm_name,
                            args=self.normalize_args(norm_name, args),
                            raw_source=body,
                        )
                    )

        return calls

    def _try_parse_xml_or_dsml(self, text: str) -> list[ToolCall]:
        """Parses DSML (<｜DSML｜invoke...>), Anthropic (<tool_use>...), and XML formats (Hermes, Claude, Kimi)."""
        if "<" not in text:
            return []

        calls: list[ToolCall] = []

        # 1. Hermes / Kimi direct tags format: <tool_name>search</tool_name><pattern>...</pattern>
        # Only check direct tags if there are no outer invoke/tool_use wrapper tags
        has_invoke_wrapper = bool(_INVOKE_RE.search(text))
        if not has_invoke_wrapper and any(
            f"<{t}>" in text.lower() for t in ("tool_name", "function_name", "tool")
        ):
            m_tname = _DIRECT_TAG_TNAME_RE.search(text)
            if m_tname:
                tname = self.normalize_name(m_tname.group(1))
                params: dict[str, Any] = {}
                tag_matches = _TAG_PAIRS_RE.finditer(text)
                for tm in tag_matches:
                    ptag = tm.group(1).lower()
                    if ptag not in ("tool_name", "function_name", "tool"):
                        val = clean_param_val(
                            tm.group(2), is_code_param=(ptag in self.code_param_keys)
                        )
                        params[ptag] = val

                if self._is_tool_allowed(tname):
                    calls.append(
                        ToolCall(
                            name=tname,
                            args=self.normalize_args(tname, params),
                            raw_source=text,
                        )
                    )
                    return calls

        # 2. Match invoke/tool/call/tool_use tags
        inv_matches = list(_INVOKE_RE.finditer(text))
        for inv in inv_matches:
            colon_tool = inv.group("colon_tool")
            attrs = inv.group("attrs") or ""
            body = inv.group("body") or ""

            tool_name = colon_tool
            if not tool_name:
                m_fn = _ATTR_TOOL_NAME_RE.search(attrs)
                if m_fn:
                    tool_name = m_fn.group(1)

            # Check child tags for tool name (Anthropic format: <name>read_file</name>)
            if not tool_name:
                m_child = _CHILD_TOOL_NAME_RE.search(body)
                if m_child:
                    tool_name = m_child.group(1)

            params = {}

            # Check DSML / standard parameter tags: <parameter name="...">
            for pname, pval in _extract_xml_parameters(body):
                is_code = pname in self.code_param_keys
                params[pname] = clean_param_val(pval, is_code_param=is_code)

            # Check for <arguments>...</arguments> or <parameters>...</parameters> block (Anthropic)
            m_wrap = _ARGUMENTS_BLOCK_RE.search(body)
            if m_wrap:
                wrap_body = m_wrap.group(1).strip()
                cand = extract_json_objects(wrap_body)
                if cand:
                    d = safe_json_loads(cand[0], default=None)
                    if isinstance(d, dict):
                        params.update(d)
                if not params:
                    tag_matches = _WRAP_TAG_MATCHES_RE.finditer(wrap_body)
                    for tm in tag_matches:
                        ptag = tm.group(1).lower()
                        if ptag not in ("arguments", "parameters"):
                            val = clean_param_val(
                                tm.group(2),
                                is_code_param=(ptag in self.code_param_keys),
                            )
                            params[ptag] = val

            # Check for direct child XML tags in body if still no params
            if not params:
                tag_matches = _TAG_PAIRS_RE.finditer(body)
                for tm in tag_matches:
                    ptag = tm.group(1).lower()
                    if ptag not in (
                        "name",
                        "tool_name",
                        "function_name",
                        "tool",
                        "function",
                        "arguments",
                        "parameters",
                    ):
                        val = clean_param_val(
                            tm.group(2), is_code_param=(ptag in self.code_param_keys)
                        )
                        params[ptag] = val

            if not tool_name:
                tool_name = (
                    params.pop("tool", None)
                    or params.pop("name", None)
                    or params.pop("action", None)
                )

            # If still no params, check for embedded JSON in body
            if not params:
                m_code = _JSON_BLOCK_RE.search(body)
                json_str = m_code.group(1) if m_code else None
                if not json_str:
                    candidates = extract_json_objects(body)
                    if candidates:
                        json_str = candidates[0]
                if json_str:
                    d = safe_json_loads(json_str, default={})
                    if isinstance(d, dict):
                        d_name, d_args = self._extract_name_and_args_from_dict(d)
                        if not tool_name and d_name:
                            tool_name = d_name
                        params.update(d_args)

            if tool_name:
                norm_name = self.normalize_name(tool_name)
                if self._is_tool_allowed(norm_name):
                    calls.append(
                        ToolCall(
                            name=norm_name,
                            args=self.normalize_args(norm_name, params),
                            raw_source=inv.group(0),
                        )
                    )

        if calls:
            return calls

        # 3. Standalone parameters directly inside <tool_calls> without invoke wrapper
        if _STANDALONE_TRIGGER_RE.search(text):
            standalone_params: dict[str, Any] = {}
            for pname, pval in _extract_xml_parameters(text):
                is_code = pname in self.code_param_keys
                standalone_params[pname] = clean_param_val(pval, is_code_param=is_code)

            tool_name = (
                standalone_params.pop("tool", None)
                or standalone_params.pop("name", None)
                or standalone_params.pop("action", None)
            )
            if not tool_name and self.allowed_tools:
                lower_text = text.lower()
                for t_tag in self.allowed_tools:
                    if t_tag.lower() not in lower_text:
                        continue
                    if re.search(rf"<[｜|]*(?:dsml[｜|]*)?{t_tag} [^>]*>", text, re.IGNORECASE):
                        tool_name = t_tag
                        break

            if tool_name:
                norm_name = self.normalize_name(tool_name)
                if self._is_tool_allowed(norm_name):
                    calls.append(
                        ToolCall(
                            name=norm_name,
                            args=self.normalize_args(norm_name, standalone_params),
                            raw_source=text,
                        )
                    )

        return calls

    def _try_parse_qwen_agent(self, text: str) -> list[ToolCall]:
        """Parses Qwen-Agent format:
        <|action_start|><|action_name|>tool_name<|action_args|>args<|action_end|>
        """
        if "action_name" not in text:
            return []

        calls: list[ToolCall] = []
        for m in _QWEN_AGENT_RE.finditer(text):
            raw_name = m.group("name").strip()
            name = self.normalize_name(raw_name)
            if not self._is_tool_allowed(name):
                continue

            args_str = m.group("args").strip()
            args: dict[str, Any] = {}
            if args_str:
                d = safe_json_loads(args_str, default=None)
                if isinstance(d, dict):
                    args = d
                elif d is not None:
                    args = {"input": d}
                else:
                    args = {"input": args_str}

            calls.append(
                ToolCall(
                    name=name,
                    args=self.normalize_args(name, args),
                    raw_source=m.group(0),
                )
            )

        return calls

    def _try_parse_chatglm(self, text: str) -> list[ToolCall]:
        """Parses ChatGLM / Legacy Qwen format:
        ✿FUNCTION✿: tool_name
        ✿ARGS✿: {"path": "foo.py"}
        """
        if "FUNCTION" not in text:
            return []

        calls: list[ToolCall] = []
        for m in _CHATGLM_RE.finditer(text):
            raw_name = m.group("name").strip()
            name = self.normalize_name(raw_name)
            if not self._is_tool_allowed(name):
                continue

            args_str = m.group("args").strip()
            args: dict[str, Any] = {}
            if args_str:
                d = safe_json_loads(args_str, default=None)
                if isinstance(d, dict):
                    args = d
                elif d is not None:
                    args = {"input": d}
                else:
                    args = {"input": args_str}

            calls.append(
                ToolCall(
                    name=name,
                    args=self.normalize_args(name, args),
                    raw_source=m.group(0),
                )
            )

        return calls

    def _try_parse_mistral(self, text: str) -> list[ToolCall]:
        """Parses Mistral [TOOL_CALLS] format:
        [TOOL_CALLS] [{"name": "search", "arguments": {"query": "foo"}}]
        """
        if "[TOOL_CALL" not in text:
            return []

        calls: list[ToolCall] = []
        for m in _MISTRAL_TOOL_CALL_RE.finditer(text):
            body = m.group("body").strip()
            data = safe_json_loads(body, default=None)
            if data is not None:
                extracted = self._extract_calls_from_dict(data, source=m.group(0))
                calls.extend(extracted)

        return calls

    def _try_parse_llama_python_tag(self, text: str) -> list[ToolCall]:
        """Parses Llama 3.1+ <|python_tag|> format:
        <|python_tag|>bash.call(command="pytest")
        or
        <|python_tag|>{"name": "bash", "parameters": {"command": "pytest"}}
        """
        if "python_tag" not in text:
            return []

        calls: list[ToolCall] = []
        for m in _LLAMA_PYTHON_TAG_RE.finditer(text):
            body = m.group("body").strip()
            # 1. Try parsing body as JSON
            sub_calls = self._try_parse_json(body)
            if sub_calls:
                for c in sub_calls:
                    c.raw_source = m.group(0)
                calls.extend(sub_calls)
                continue

            # 2. Try parsing body as Python call
            sub_calls = self._try_parse_python_expr(body)
            if sub_calls:
                for c in sub_calls:
                    c.raw_source = m.group(0)
                calls.extend(sub_calls)

        return calls

    def _try_parse_python_expr(self, text: str) -> list[ToolCall]:
        """Parses Python function call expressions like read_file(path="foo.py", offset=10)
        or bash.call(command="pytest") using standard library ast.
        """
        if "(" not in text:
            return []

        candidate = text.strip()
        # Strip markdown ```python ... ``` if wrapped
        if candidate.startswith("```"):
            m = _PYTHON_CODEBLOCK_RE.search(candidate)
            if m:
                candidate = m.group(1).strip()

        calls: list[ToolCall] = []

        # Try parsing entire candidate as AST module
        cand_clean = candidate
        if cand_clean.startswith("call:"):
            cand_clean = cand_clean[5:].strip()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(cand_clean)
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call = self._ast_call_to_tool_call(node.value, source=candidate)
                    if call:
                        calls.append(call)
            if calls:
                return calls
        except (SyntaxError, ValueError, TypeError):
            pass

        # If full parse failed (e.g. conversational text with python calls), parse line by line
        lines = [candidate] if "\n" not in candidate else candidate.splitlines()
        for line in lines:
            orig_line = line.strip()
            line_clean = orig_line
            if line_clean.startswith("call:"):
                line_clean = line_clean[5:].strip()
            if not line_clean or line_clean.startswith("#") or "(" not in line_clean:
                continue

            call_node = None
            for suffix in ("", ")", '")', '"))', "}"):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tree = ast.parse(line_clean + suffix)
                    if (
                        tree.body
                        and isinstance(tree.body[0], ast.Expr)
                        and isinstance(tree.body[0].value, ast.Call)
                    ):
                        call_node = tree.body[0].value
                        break
                except (SyntaxError, ValueError, TypeError):
                    continue

            if call_node is None:
                continue

            call = self._ast_call_to_tool_call(call_node, source=orig_line)
            if call:
                calls.append(call)

        return calls

    def _ast_call_to_tool_call(self, call_node: ast.Call, source: str = "") -> ToolCall | None:
        raw_name = self._extract_ast_func_name(call_node.func)
        if not raw_name:
            return None

        if self.allowed_tools is None and raw_name.lower() in _PYTHON_BUILTINS_IGNORE:
            return None

        name = self.normalize_name(raw_name)
        if not self._is_tool_allowed(name):
            return None

        args: dict[str, Any] = {}
        for kw in call_node.keywords:
            if kw.arg:
                try:
                    val = ast.literal_eval(kw.value)
                except (
                    ValueError,
                    SyntaxError,
                    TypeError,
                    MemoryError,
                    RecursionError,
                ):
                    try:
                        val = ast.unparse(kw.value)
                    except (ValueError, TypeError, AttributeError):
                        val = ""
                args[kw.arg] = val

        for idx, p_arg in enumerate(call_node.args):
            try:
                val = ast.literal_eval(p_arg)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                try:
                    val = ast.unparse(p_arg)
                except (ValueError, TypeError, AttributeError):
                    val = ""
            args[f"arg{idx}"] = val

        return ToolCall(
            name=name,
            args=self.normalize_args(name, args),
            raw_source=source,
        )

    def _extract_ast_func_name(self, func_node: ast.AST) -> str | None:
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            if func_node.attr in ("call", "run", "execute", "invoke") and isinstance(
                func_node.value, ast.Name
            ):
                return func_node.value.id
            elif isinstance(func_node.value, ast.Name) and func_node.value.id in (
                "tool",
                "tools",
                "action",
                "functions",
            ):
                return func_node.attr
            else:
                return func_node.attr
        return None

    def _try_parse_react(self, text: str) -> list[ToolCall]:
        """Parses ReAct format:
        Action: bash
        Action Input: pytest -v
        """
        if "Action:" not in text and "action:" not in text:
            return []

        calls: list[ToolCall] = []
        for m in _REACT_RE.finditer(text):
            raw_name = m.group("name").strip()
            name = self.normalize_name(raw_name)
            if not self._is_tool_allowed(name):
                continue

            action_input = m.group("args").strip()
            args: dict[str, Any] = {}

            # Strip markdown fence if present
            m_fence = _FENCE_CODEBLOCK_RE.search(action_input)
            if m_fence:
                action_input = m_fence.group(1).strip()

            # Try parsing action_input as JSON
            cand = extract_json_objects(action_input)
            if cand:
                d = safe_json_loads(cand[0], default=None)
                if isinstance(d, dict):
                    args = d

            if not args:
                if name == "bash":
                    args = {"command": action_input}
                elif name in ("read_file", "view_file", "cat"):
                    args = {"path": action_input}
                else:
                    args = {"input": action_input}

            calls.append(
                ToolCall(
                    name=name,
                    args=self.normalize_args(name, args),
                    raw_source=m.group(0),
                )
            )

        return calls

    def _repair_headless_json(self, raw: str) -> list[str]:
        """Reconstructs prefix-truncated / headless JSON objects (e.g. MiniMax prefill drops)."""
        cands: list[str] = []

        # 1. Colon prefix omission: `: "tool_name", ...` or `": "tool_name", ...`
        m1 = _HEADLESS_COLON_RE.search(raw)
        if m1:
            tname = m1.group("tname")
            rest = m1.group("rest")
            cands.append(f'{{"tool": "{tname}", {rest}')
            cands.append(f'{{"name": "{tname}", {rest}')

        # 2. Unquoted or unbraced key omission: `tool": "bash", ...` or `"tool": "bash", ...`
        m2 = _HEADLESS_KEY_RE.search(raw)
        if m2:
            open_q = m2.group("open")
            close_q = m2.group("close") or '"'
            key = m2.group("key")
            rest = m2.group("rest")
            if open_q is None:
                cands.append(f"{{{close_q}{key}{close_q}: {rest}")
            else:
                start_idx = m2.start("open")
                cands.append("{" + raw[start_idx:])

        return cands

    def _try_parse_json(self, raw: str) -> list[ToolCall]:
        """Extracts tool calls formatted as JSON (markdown blocks, OpenAI formats, conversational JSON)."""
        if "{" not in raw and "[" not in raw and "```" not in raw and ":" not in raw:
            return []

        calls: list[ToolCall] = []

        # =====================================================================
        # TIER 1: STRICT SPECIFICATION COMPLIANCE (Fast-Path)
        # Checks canonical markdown JSON code blocks, RFC-8259 arrays, and balanced objects.
        # =====================================================================

        # 1. Check all markdown code blocks (```json ... ```)
        m_blocks = list(_JSON_BLOCK_RE.finditer(raw)) + list(_JSON_ARRAY_BLOCK_RE.finditer(raw))
        if m_blocks:
            for mb in m_blocks:
                blob = mb.group(1)
                data = safe_json_loads(blob, default=None)
                if data is not None:
                    extracted = self._extract_calls_from_dict(data, source=blob)
                    calls.extend(extracted)
            if calls:
                return calls

        # 2. Check if raw text is a top-level JSON array
        trimmed = raw.strip()
        if trimmed.startswith("[") and trimmed.endswith("]"):
            data = safe_json_loads(trimmed, default=None)
            if data is not None:
                extracted = self._extract_calls_from_dict(data, source=trimmed)
                if extracted:
                    return extracted

        # 3. Check JSON candidates via extract_json_objects (balanced braces)
        candidates = extract_json_objects(raw)
        if candidates:
            for cand in candidates:
                data = safe_json_loads(cand, default=None)
                if data is not None:
                    extracted = self._extract_calls_from_dict(data, source=cand)
                    calls.extend(extracted)
        if calls:
            return calls

        # =====================================================================
        # TIER 2: FAULT-TOLERANT RECOVERY (ParseGuard Layer)
        # If input deviates from balanced JSON specification (e.g. MiniMax / prompt-prefill
        # prefix omissions: `tool": "bash", ...` or `: "read_file", ...`), attempt deterministic recovery.
        # =====================================================================
        recovered = self._repair_headless_json(raw)
        for cand in recovered:
            sub_candidates = extract_json_objects(cand)
            for sub_cand in sub_candidates:
                data = safe_json_loads(sub_cand, default=None)
                if data is not None:
                    extracted = self._extract_calls_from_dict(data, source=sub_cand)
                    calls.extend(extracted)
            if calls:
                return calls

        return calls

    def _try_parse_shell(self, raw: str) -> list[ToolCall]:
        """Fallback for command lines starting with $, #, python, pytest, git."""
        if not self.allow_shell_fallback:
            return []

        calls: list[ToolCall] = []
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        for line in lines:
            if line.startswith(("$ ", "# ")):
                cmd = line[2:].strip()
                calls.append(ToolCall(name="bash", args={"command": cmd}, raw_source=line))
            elif line.startswith(("python ", "git ", "pytest ")):
                calls.append(ToolCall(name="bash", args={"command": line}, raw_source=line))
            elif line.lower() == "submit":
                calls.append(ToolCall(name="submit", args={}, raw_source=line))

        return calls

    def parse_all(self, text: str) -> list[ToolCall]:
        """Parses an LLM response string and returns all structured ToolCalls found.

        Raises:
            ToolError: If no valid tool call could be extracted or tool is not allowed.
        """
        raw = (text or "").strip()
        target = strip_thinking(raw) if self.strip_thinking else raw

        # 1. Native DeepSeek special tokens format
        calls = self._try_parse_deepseek_native(target)
        if calls:
            return calls

        # 2. Qwen-Agent tokens
        calls = self._try_parse_qwen_agent(target)
        if calls:
            return calls

        # 3. Mistral [TOOL_CALLS] tokens
        calls = self._try_parse_mistral(target)
        if calls:
            return calls

        # 4. Llama 3.1+ <|python_tag|>
        calls = self._try_parse_llama_python_tag(target)
        if calls:
            return calls

        # 5. XML / DSML / Anthropic <tool_use> / Hermes <tool_call>
        calls = self._try_parse_xml_or_dsml(target)
        if calls:
            return calls

        # 6. ChatGLM / Legacy Qwen ✿FUNCTION✿
        calls = self._try_parse_chatglm(target)
        if calls:
            return calls

        # 7. ReAct Action: / Action Input:
        calls = self._try_parse_react(target)
        if calls:
            return calls

        # 8. JSON (OpenAI formats, markdown blocks, conversational inline JSON)
        calls = self._try_parse_json(target)
        if calls:
            return calls

        # 9. Python function call syntax (read_file(...) or bash.call(...))
        calls = self._try_parse_python_expr(target)
        if calls:
            return calls

        # 10. Optional shell fallback
        calls = self._try_parse_shell(raw)
        if calls:
            return calls

        preview = raw[:200] + ("..." if len(raw) > 200 else "")
        raise ToolError(f"No recognized tool call found in LLM response: {preview!r}")

    parse_many = parse_all
    parse_calls = parse_all

    def parse(self, text: str) -> ToolCall:
        """Parses an LLM response string and returns the first structured ToolCall.

        Raises:
            ToolError: If no valid tool call could be extracted or tool is not allowed.
        """
        calls = self.parse_all(text)
        return calls[0]

    def try_parse_all(self, text: str) -> list[ToolCall]:
        """Attempts to parse all tool calls, returning empty list on failure instead of raising."""
        try:
            return self.parse_all(text)
        except ToolError:
            return []

    try_parse_many = try_parse_all
    try_parse_calls = try_parse_all

    def try_parse(self, text: str) -> ToolCall | None:
        """Attempts to parse tool call, returning None on failure instead of raising."""
        try:
            return self.parse(text)
        except ToolError:
            return None


def parse_tool_call(
    text: str,
    allowed_tools: Collection[str] | None = None,
    tool_aliases: dict[str, str] | None = None,
    param_aliases: dict[str, str] | None = None,
    code_param_keys: Collection[str] | None = None,
    strip_thinking: bool = True,
    allow_shell_fallback: bool = False,
) -> ToolCall:
    """Convenience function to parse an LLM response with custom or default settings."""
    parser = ToolParser(
        allowed_tools=allowed_tools,
        tool_aliases=tool_aliases,
        param_aliases=param_aliases,
        code_param_keys=code_param_keys,
        strip_thinking=strip_thinking,
        allow_shell_fallback=allow_shell_fallback,
    )
    return parser.parse(text)


def try_parse_tool_call(
    text: str,
    allowed_tools: Collection[str] | None = None,
    tool_aliases: dict[str, str] | None = None,
    param_aliases: dict[str, str] | None = None,
    code_param_keys: Collection[str] | None = None,
    strip_thinking: bool = True,
    allow_shell_fallback: bool = False,
) -> ToolCall | None:
    """Convenience function to parse an LLM response, returning None instead of raising."""
    parser = ToolParser(
        allowed_tools=allowed_tools,
        tool_aliases=tool_aliases,
        param_aliases=param_aliases,
        code_param_keys=code_param_keys,
        strip_thinking=strip_thinking,
        allow_shell_fallback=allow_shell_fallback,
    )
    return parser.try_parse(text)


def parse_tool_calls(
    text: str,
    allowed_tools: Collection[str] | None = None,
    tool_aliases: dict[str, str] | None = None,
    param_aliases: dict[str, str] | None = None,
    code_param_keys: Collection[str] | None = None,
    strip_thinking: bool = True,
    allow_shell_fallback: bool = False,
) -> list[ToolCall]:
    """Parses an LLM response string and returns all structured ToolCalls found."""
    parser = ToolParser(
        allowed_tools=allowed_tools,
        tool_aliases=tool_aliases,
        param_aliases=param_aliases,
        code_param_keys=code_param_keys,
        strip_thinking=strip_thinking,
        allow_shell_fallback=allow_shell_fallback,
    )
    return parser.parse_all(text)


def try_parse_tool_calls(
    text: str,
    allowed_tools: Collection[str] | None = None,
    tool_aliases: dict[str, str] | None = None,
    param_aliases: dict[str, str] | None = None,
    code_param_keys: Collection[str] | None = None,
    strip_thinking: bool = True,
    allow_shell_fallback: bool = False,
) -> list[ToolCall]:
    """Attempts to parse all tool calls, returning empty list on failure instead of raising."""
    parser = ToolParser(
        allowed_tools=allowed_tools,
        tool_aliases=tool_aliases,
        param_aliases=param_aliases,
        code_param_keys=code_param_keys,
        strip_thinking=strip_thinking,
        allow_shell_fallback=allow_shell_fallback,
    )
    return parser.try_parse_all(text)
