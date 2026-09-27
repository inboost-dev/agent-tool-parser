# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Represents a structured tool invocation extracted from LLM output.

    Attributes:
        name: Normalized name of the tool to execute.
        args: Dictionary of parsed arguments/parameters for the tool.
        raw_source: Snippet of the original response that matched.
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    raw_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "raw_source": self.raw_source,
        }

    def to_canonical_json(self) -> str:
        """Serialize tool arguments into deterministic, sorted-key JSON string for KV-cache invariance."""
        from agent_tool_parser.canonical import canonical_json_dumps

        return canonical_json_dumps(self.args)

    def to_openai_tool_call(
        self, call_id: str | None = None, canonical: bool = True
    ) -> dict[str, Any]:
        """Convert ToolCall into standard OpenAI tool_call schema object."""
        import json

        arguments_str = self.to_canonical_json() if canonical else json.dumps(self.args)
        return {
            "id": call_id or f"call_{self.name}",
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": arguments_str,
            },
        }

    def __repr__(self) -> str:
        return f"ToolCall(name={self.name!r}, args={self.args!r})"


class ToolError(ValueError):
    """Raised when no valid tool call could be extracted or validation failed."""
