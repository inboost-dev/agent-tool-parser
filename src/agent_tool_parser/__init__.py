"""agent-tool-parser: Fast, multi-format parser for LLM agent tool calls.

Copyright (c) 2026 InBoost Team.
Licensed under the MIT License. Provided "AS IS" without warranty of any kind.
See LICENSE and README.md for full terms and execution safety disclaimers.
"""

from __future__ import annotations

import os

from agent_tool_parser.canonical import (
    canonical_json_dumps as _py_canonical_json_dumps,
)
from agent_tool_parser.canonical import (
    canonical_sort_keys as _py_canonical_sort_keys,
)
from agent_tool_parser.canonical import (
    canonicalize_arguments_string as _py_canonicalize_arguments_string,
)
from agent_tool_parser.canonical import (
    canonicalize_tool_calls,
)
from agent_tool_parser.cleaners import clean_param_val
from agent_tool_parser.raw import (
    RawToolCall as _PyRawToolCall,
)
from agent_tool_parser.raw import (
    extract_raw_tool_calls as _py_extract_raw_tool_calls,
)
from agent_tool_parser.raw import (
    try_extract_raw_tool_call as _py_try_extract_raw_tool_call,
)
from agent_tool_parser.streaming import (
    StreamEvent as _PyStreamEvent,
)
from agent_tool_parser.streaming import (
    StreamingToolParser as _PyStreamingToolParser,
)

# Allow forcing pure-Python mode via environment variable for benchmarking or debugging
_DISABLE_ACCEL = os.getenv("AGENT_TOOL_PARSER_NO_EXT", "0") in ("1", "true", "True")

ACCELERATED: bool = False

if not _DISABLE_ACCEL:
    try:
        from agent_tool_parser._accelerated import (  # type: ignore[import-untyped,import-not-found]
            RawToolCall,
            StreamEvent,
            StreamingToolParser,
            ToolCall,
            ToolError,
            ToolParser,
            canonical_json_dumps,
            canonical_sort_keys,
            canonicalize_arguments_string,
            clean_json_str,
            extract_json_objects,
            extract_raw_tool_calls,
            normalize_truncated_json_prefix,
            parse_tool_call,
            parse_tool_calls,
            safe_json_loads,
            strip_thinking,
            try_extract_raw_tool_call,
            try_parse_tool_call,
            try_parse_tool_calls,
        )

        ACCELERATED = True
    except ImportError:
        pass

if not ACCELERATED:
    from agent_tool_parser.cleaners import (
        clean_json_str,
        extract_json_objects,
        normalize_truncated_json_prefix,
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

    canonical_json_dumps = _py_canonical_json_dumps
    canonical_sort_keys = _py_canonical_sort_keys
    canonicalize_arguments_string = _py_canonicalize_arguments_string
    RawToolCall = _PyRawToolCall  # type: ignore[misc,assignment]
    extract_raw_tool_calls = _py_extract_raw_tool_calls
    try_extract_raw_tool_call = _py_try_extract_raw_tool_call
    StreamEvent = _PyStreamEvent  # type: ignore[misc,assignment]
    StreamingToolParser = _PyStreamingToolParser  # type: ignore[misc,assignment]

__version__ = "0.1.4-dev"
__all__ = [
    "ACCELERATED",
    "RawToolCall",
    "StreamEvent",
    "StreamingToolParser",
    "ToolCall",
    "ToolError",
    "ToolParser",
    "canonical_json_dumps",
    "canonical_sort_keys",
    "canonicalize_arguments_string",
    "canonicalize_tool_calls",
    "clean_json_str",
    "clean_param_val",
    "extract_json_objects",
    "extract_raw_tool_calls",
    "normalize_truncated_json_prefix",
    "parse_tool_call",
    "parse_tool_calls",
    "safe_json_loads",
    "strip_thinking",
    "try_extract_raw_tool_call",
    "try_parse_tool_call",
    "try_parse_tool_calls",
]
