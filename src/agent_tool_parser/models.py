# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_tool_parser.semantics import (
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

    @property
    def category(self) -> ToolCategory:
        """Semantic category of this tool call."""
        return categorize_tool(self.name, self.args)

    @property
    def normalized_name(self) -> str:
        """Canonicalized lowercase tool name."""
        return normalize_tool_name(self.name)

    @property
    def normalized_args(self) -> dict[str, Any]:
        """Dictionary of arguments with parameter aliases resolved."""
        return normalize_tool_args(self.args)

    @property
    def is_mutation(self) -> bool:
        """Whether this tool modifies files or code on disk."""
        return is_mutation_tool(self.name, self.args)

    @property
    def is_verification(self) -> bool:
        """Whether this tool executes test suites or verification checks."""
        return is_verification_tool(self.name, self.args)

    @property
    def is_execution(self) -> bool:
        """Whether this tool executes system shell commands."""
        return is_execution_tool(self.name)

    @property
    def is_read(self) -> bool:
        """Whether this tool is an inspection or reading tool."""
        return is_read_tool(self.name)

    @property
    def is_search(self) -> bool:
        """Whether this tool searches the codebase or directory tree."""
        return is_search_tool(self.name)

    @property
    def is_submit(self) -> bool:
        """Whether this tool submits or concludes the task."""
        return is_submit_tool(self.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "raw_source": self.raw_source,
            "category": self.category.value,
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
        return f"ToolCall(name={self.name!r}, category={self.category.value!r}, args={self.args!r})"


class ToolError(ValueError):
    """Raised when no valid tool call could be extracted or validation failed."""
