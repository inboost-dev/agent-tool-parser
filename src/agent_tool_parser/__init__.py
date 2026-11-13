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
from agent_tool_parser.python_calls import extract_python_function_calls
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
from agent_tool_parser.vector import (
    CpuVectorEngine,
    detect_vector_engine,
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
            repair_unescaped_quotes,
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
        repair_unescaped_quotes,
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

# Universal Protocol Intermediate Representation (IR) & Adapters
from agent_tool_parser.gemini import (  # noqa: E402
    GeminiToolAdapter,
    anthropic_tools_to_gemini,
    gemini_to_anthropic_tool_use_blocks,
    gemini_to_openai_tool_calls,
    openai_tools_to_gemini,
    tool_call_to_gemini_part,
    tool_result_to_gemini_part,
)

# Import Semantic Tool Categorization & Normalization
from agent_tool_parser.semantics import (  # noqa: E402
    ARG_ALIASES,
    BASH_MUTATION_OPS,
    CANONICAL_TOOL_MAP,
    TEST_CMD_PATTERNS,
    ToolCategory,
    categorize_tool,
    is_execution_tool,
    is_mutation_tool,
    is_read_tool,
    is_search_tool,
    is_submit_tool,
    is_verification_tool,
    normalize_tool_args,
    normalize_tool_name,
)
from agent_tool_parser.universal import (  # noqa: E402
    DEFAULT_PROVIDER_ALIASES,
    AnthropicProtocolAdapter,
    BaseProtocolAdapter,
    CanonicalChunk,
    CanonicalContentPart,
    CanonicalDeltaToolCall,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalTool,
    CanonicalToolCall,
    CanonicalToolResult,
    GeminiProtocolAdapter,
    OpenAIProtocolAdapter,
    ProtocolRegistry,
    StreamingProtocolTransformer,
    StreamingState,
    SupportedProtocol,
    ToolDefinition,
    ToolMetadata,
    ToolResult,
    UniversalProtocolAdapter,
    convert_tool_calls,
    convert_tool_definitions,
    convert_tool_results,
    create_streaming_transformer,
    normalize_to_ir,
    sanitize_tool_schema_for_gemini,
    sanitize_tool_schema_for_openai,
    to_mcp_call_result,
    to_mcp_tool_call,
    translate_request,
    translate_response,
)

# Bind semantic properties to ToolCall
if not hasattr(ToolCall, "category"):
    ToolCall.category = property(
        lambda self: categorize_tool(self.name, getattr(self, "args", None))
    )
if not hasattr(ToolCall, "normalized_name"):
    ToolCall.normalized_name = property(lambda self: normalize_tool_name(self.name))
if not hasattr(ToolCall, "normalized_args"):
    ToolCall.normalized_args = property(lambda self: normalize_tool_args(getattr(self, "args", {})))
if not hasattr(ToolCall, "is_mutation"):
    ToolCall.is_mutation = property(
        lambda self: is_mutation_tool(self.name, getattr(self, "args", None))
    )
if not hasattr(ToolCall, "is_verification"):
    ToolCall.is_verification = property(
        lambda self: is_verification_tool(self.name, getattr(self, "args", None))
    )
if not hasattr(ToolCall, "is_execution"):
    ToolCall.is_execution = property(lambda self: is_execution_tool(self.name))
if not hasattr(ToolCall, "is_read"):
    ToolCall.is_read = property(lambda self: is_read_tool(self.name))
if not hasattr(ToolCall, "is_search"):
    ToolCall.is_search = property(lambda self: is_search_tool(self.name))
if not hasattr(ToolCall, "is_submit"):
    ToolCall.is_submit = property(lambda self: is_submit_tool(self.name))

__version__ = "0.2.0"
__all__ = [
    "ACCELERATED",
    "ARG_ALIASES",
    "BASH_MUTATION_OPS",
    "CANONICAL_TOOL_MAP",
    "DEFAULT_PROVIDER_ALIASES",
    "TEST_CMD_PATTERNS",
    "AnthropicProtocolAdapter",
    "BaseProtocolAdapter",
    "CanonicalChunk",
    "CanonicalContentPart",
    "CanonicalDeltaToolCall",
    "CanonicalMessage",
    "CanonicalRequest",
    "CanonicalResponse",
    "CanonicalTool",
    "CanonicalToolCall",
    "CanonicalToolResult",
    "CpuVectorEngine",
    "GeminiProtocolAdapter",
    "GeminiToolAdapter",
    "OpenAIProtocolAdapter",
    "ProtocolRegistry",
    "RawToolCall",
    "StreamEvent",
    "StreamingProtocolTransformer",
    "StreamingState",
    "StreamingToolParser",
    "SupportedProtocol",
    "ToolCall",
    "ToolCategory",
    "ToolDefinition",
    "ToolError",
    "ToolMetadata",
    "ToolParser",
    "ToolResult",
    "UniversalProtocolAdapter",
    "anthropic_tools_to_gemini",
    "canonical_json_dumps",
    "canonical_sort_keys",
    "canonicalize_arguments_string",
    "canonicalize_tool_calls",
    "categorize_tool",
    "clean_json_str",
    "clean_param_val",
    "convert_tool_calls",
    "convert_tool_definitions",
    "convert_tool_results",
    "create_streaming_transformer",
    "detect_vector_engine",
    "extract_json_objects",
    "extract_python_function_calls",
    "extract_raw_tool_calls",
    "gemini_to_anthropic_tool_use_blocks",
    "gemini_to_openai_tool_calls",
    "is_execution_tool",
    "is_mutation_tool",
    "is_read_tool",
    "is_search_tool",
    "is_submit_tool",
    "is_verification_tool",
    "normalize_to_ir",
    "normalize_tool_args",
    "normalize_tool_name",
    "normalize_truncated_json_prefix",
    "openai_tools_to_gemini",
    "parse_tool_call",
    "parse_tool_calls",
    "repair_unescaped_quotes",
    "safe_json_loads",
    "sanitize_tool_schema_for_gemini",
    "sanitize_tool_schema_for_openai",
    "strip_thinking",
    "to_mcp_call_result",
    "to_mcp_tool_call",
    "tool_call_to_gemini_part",
    "tool_result_to_gemini_part",
    "translate_request",
    "translate_response",
    "try_extract_raw_tool_call",
    "try_parse_tool_call",
    "try_parse_tool_calls",
]
