"""agent-tool-parser: Fast, multi-format parser for LLM agent tool calls.

Copyright (c) 2026 InBoost Team.
Licensed under the MIT License. Provided "AS IS" without warranty of any kind.
See LICENSE and README.md for full terms and execution safety disclaimers.
"""

from __future__ import annotations

import os

# Allow forcing pure-Python mode via environment variable for benchmarking or debugging
_DISABLE_ACCEL = os.getenv("AGENT_TOOL_PARSER_NO_EXT", "0") in ("1", "true", "True")

ACCELERATED: bool = False

if not _DISABLE_ACCEL:
    try:
        from agent_tool_parser._accelerated import (  # type: ignore[import-untyped,import-not-found]
            ToolCall,
            ToolError,
            ToolParser,
            clean_json_str,
            parse_tool_call,
            parse_tool_calls,
            safe_json_loads,
            strip_thinking,
            try_parse_tool_call,
            try_parse_tool_calls,
        )

        ACCELERATED = True
    except ImportError:
        pass

if not ACCELERATED:
    from agent_tool_parser.cleaners import (
        clean_json_str,
        clean_param_val,
        extract_json_objects,
        safe_json_loads,
        strip_thinking,
    )
    from agent_tool_parser.models import ToolCall, ToolError
    from agent_tool_parser.parser import (
        ToolParser,
        parse_tool_call,
        parse_tool_calls,
        try_parse_tool_call,
        try_parse_tool_calls,
    )
else:
    from agent_tool_parser.cleaners import clean_param_val, extract_json_objects

__version__ = "0.1.0"
__all__ = [
    "ACCELERATED",
    "ToolCall",
    "ToolError",
    "ToolParser",
    "clean_json_str",
    "clean_param_val",
    "extract_json_objects",
    "parse_tool_call",
    "parse_tool_calls",
    "safe_json_loads",
    "strip_thinking",
    "try_parse_tool_call",
    "try_parse_tool_calls",
]
