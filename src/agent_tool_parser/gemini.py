"""Gemini Tool Call and Function Declaration adapter for agent-tool-parser.

Provides bidirectional translation between Google Gemini function declarations/calls,
OpenAI tool calls, and Anthropic tool use schemas.
"""

from __future__ import annotations

import json
import time
from typing import Any


def gemini_to_openai_tool_calls(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Gemini functionCall parts into OpenAI-compatible tool_calls list."""
    tool_calls: list[dict[str, Any]] = []
    idx = 0
    for part in parts:
        if not isinstance(part, dict):
            continue
        fc = part.get("functionCall")
        if fc and isinstance(fc, dict):
            name = fc.get("name", "tool")
            args = fc.get("args", {})
            args_str = json.dumps(args) if isinstance(args, dict) else str(args)
            tool_calls.append(
                {
                    "id": f"call_{name}_{idx}_{int(time.time())}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str,
                    },
                }
            )
            idx += 1
    return tool_calls


def gemini_to_anthropic_tool_use_blocks(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Gemini functionCall parts into Anthropic tool_use content blocks."""
    blocks: list[dict[str, Any]] = []
    idx = 0
    for part in parts:
        if not isinstance(part, dict):
            continue
        fc = part.get("functionCall")
        if fc and isinstance(fc, dict):
            name = fc.get("name", "tool")
            args = fc.get("args", {})
            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{name}_{idx}_{int(time.time())}",
                    "name": name,
                    "input": args if isinstance(args, dict) else {},
                }
            )
            idx += 1
    return blocks


def openai_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tools schema to Gemini function_declarations format."""
    declarations: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            declarations.append(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        elif "name" in t:
            # Direct function definition
            declarations.append(
                {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                }
            )
    if declarations:
        return [{"function_declarations": declarations}]
    return []


def anthropic_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tools schema to Gemini function_declarations format."""
    declarations: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if name:
            declarations.append(
                {
                    "name": name,
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                }
            )
    if declarations:
        return [{"function_declarations": declarations}]
    return []


def tool_call_to_gemini_part(name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
    """Create a Gemini functionCall part from a name and argument dict/JSON string."""
    if isinstance(arguments, str):
        try:
            parsed_args = json.loads(arguments)
        except Exception:
            parsed_args = {"raw": arguments}
    else:
        parsed_args = arguments or {}
    return {"functionCall": {"name": name, "args": parsed_args}}


def tool_result_to_gemini_part(name: str, content: Any) -> dict[str, Any]:
    """Create a Gemini functionResponse part from tool result."""
    response_payload = content if isinstance(content, (dict, list)) else {"text": str(content)}
    return {
        "functionResponse": {
            "name": name,
            "response": {"content": response_payload},
        }
    }


class GeminiToolAdapter:
    """Consolidated adapter for Gemini tool schema and call conversions."""

    to_openai_tool_calls = staticmethod(gemini_to_openai_tool_calls)
    to_anthropic_tool_use = staticmethod(gemini_to_anthropic_tool_use_blocks)
    openai_tools_to_gemini = staticmethod(openai_tools_to_gemini)
    anthropic_tools_to_gemini = staticmethod(anthropic_tools_to_gemini)
    call_to_part = staticmethod(tool_call_to_gemini_part)
    result_to_part = staticmethod(tool_result_to_gemini_part)
