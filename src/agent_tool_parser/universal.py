"""Universal Any-to-Any Protocol Adapter and Canonical IR for AI Agents and LLMs.

Enables seamless N-to-N bidirectional translation between:
- OpenAI Chat Completions (Cursor, Codex, OpenHands, Cline, vLLM, SGLang, Ollama)
- Anthropic Messages (Claude Code, Anthropic SDK)
- Google Gemini Contents (Antigravity, Gemini API, Code Assist)
- Open-weights Tool Calling (DeepSeek, Qwen XML, Mistral)

Uses a Canonical Intermediate Representation (IR) to achieve O(N) complexity
instead of O(N^2) pairwise adapters, with full streaming SSE transformation support.
"""

from __future__ import annotations

import abc
import copy
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

SupportedProtocol = Literal["openai", "anthropic", "gemini"]


# ============================================================================
# 1. CANONICAL INTERMEDIATE REPRESENTATION (IR) DATA STRUCTURES
# ============================================================================


@dataclass
class CanonicalToolCall:
    """Represents a discrete tool/function call emitted by an LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass
class CanonicalToolResult:
    """Represents the execution outcome of a tool call returned to the LLM."""

    tool_use_id: str
    content: str
    name: str = ""
    is_error: bool = False


@dataclass
class CanonicalContentPart:
    """Represents a multimodal or typed content block (text, image, thinking)."""

    type: Literal["text", "image", "thinking"] = "text"
    text: str | None = None
    media_type: str | None = None
    data: str | None = None  # Base64 data
    url: str | None = None


@dataclass
class CanonicalMessage:
    """Represents a single conversational turn in the canonical sequence."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    parts: list[CanonicalContentPart] = field(default_factory=list)
    tool_calls: list[CanonicalToolCall] = field(default_factory=list)
    tool_results: list[CanonicalToolResult] = field(default_factory=list)
    thinking: str | None = None


@dataclass
class CanonicalTool:
    """Represents a tool / function specification declared to the LLM."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalRequest:
    """Standardized representation of an LLM generation request."""

    model: str
    messages: list[CanonicalMessage] = field(default_factory=list)
    system: str | None = None
    tools: list[CanonicalTool] = field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False
    thinking_budget: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalResponse:
    """Standardized representation of a non-streaming LLM generation response."""

    content: str | None
    model: str
    tool_calls: list[CanonicalToolCall] = field(default_factory=list)
    thinking: str | None = None
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: dict[str, Any] | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalDeltaToolCall:
    """Represents an incremental tool call delta in a streaming chunk."""

    index: int = 0
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


@dataclass
class CanonicalChunk:
    """Standardized representation of an incremental streaming event."""

    text_delta: str | None = None
    thinking_delta: str | None = None
    tool_call_deltas: list[CanonicalDeltaToolCall] = field(default_factory=list)
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_chunk: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 2. STREAMING STATE & ABSTRACT PROTOCOL ADAPTER INTERFACE
# ============================================================================


@dataclass
class StreamingState:
    """Encapsulates streaming session state across incremental chunks."""

    stream_id: str
    model: str
    source_protocol: str
    target_protocol: str
    # Index tracking
    tool_call_indices: dict[str, int] = field(default_factory=dict)
    next_tool_index: int = 0
    current_block_index: int = 0
    # Anthropic event lifecycle flags
    sent_message_start: bool = False
    active_block_type: str | None = None
    active_block_index: int = 0
    # Accumulation buffers
    accumulated_arguments: dict[int, str] = field(default_factory=dict)
    accumulated_text: str = ""
    accumulated_thinking: str = ""


class BaseProtocolAdapter(abc.ABC):
    """Abstract Strategy interface for protocol-specific serialization & parsing."""

    protocol_name: str

    @abc.abstractmethod
    def parse_request(self, payload: dict[str, Any]) -> CanonicalRequest:
        """Parse native request payload into CanonicalRequest IR."""
        ...

    @abc.abstractmethod
    def render_request(self, req: CanonicalRequest) -> dict[str, Any]:
        """Render CanonicalRequest IR into native request payload."""
        ...

    @abc.abstractmethod
    def parse_response(
        self, payload: dict[str, Any], model: str | None = None
    ) -> CanonicalResponse:
        """Parse native response payload into CanonicalResponse IR."""
        ...

    @abc.abstractmethod
    def render_response(self, resp: CanonicalResponse) -> dict[str, Any]:
        """Render CanonicalResponse IR into native response payload."""
        ...

    @abc.abstractmethod
    def parse_chunk(self, chunk: dict[str, Any], state: StreamingState) -> CanonicalChunk:
        """Parse native streaming chunk into CanonicalChunk IR."""
        ...

    @abc.abstractmethod
    def render_chunk(self, chunk: CanonicalChunk, state: StreamingState) -> list[dict[str, Any]]:
        """Render CanonicalChunk IR into native streaming chunks / events."""
        ...


# ============================================================================
# 3. CONCRETE PROTOCOL ADAPTERS: OPENAI, ANTHROPIC, GEMINI
# ============================================================================


class OpenAIProtocolAdapter(BaseProtocolAdapter):
    """Adapter for OpenAI Chat Completions API (/v1/chat/completions)."""

    protocol_name = "openai"

    def parse_request(self, payload: dict[str, Any]) -> CanonicalRequest:
        req = CanonicalRequest(
            model=payload.get("model", "default-model"),
            stream=payload.get("stream", False),
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
            max_tokens=payload.get("max_tokens"),
        )
        stops = payload.get("stop")
        if stops:
            req.stop_sequences = [stops] if isinstance(stops, str) else list(stops)

        # Tools
        for t in payload.get("tools", []):
            if t.get("type") == "function" and "function" in t:
                fn = t["function"]
                req.tools.append(
                    CanonicalTool(
                        name=fn.get("name", ""),
                        description=fn.get("description", ""),
                        parameters=fn.get("parameters", {}),
                    )
                )

        # Messages
        for msg in payload.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            thinking = msg.get("reasoning_content") or msg.get("thinking")

            if role == "system":
                sys_str = content if isinstance(content, str) else json.dumps(content)
                if req.system:
                    req.system += "\n\n" + sys_str
                else:
                    req.system = sys_str
            elif role == "user":
                parts: list[CanonicalContentPart] = []
                text_content = ""
                if isinstance(content, str):
                    text_content = content
                    parts.append(CanonicalContentPart(type="text", text=content))
                elif isinstance(content, list):
                    text_fragments = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                txt = item.get("text", "")
                                text_fragments.append(txt)
                                parts.append(CanonicalContentPart(type="text", text=txt))
                            elif item.get("type") == "image_url":
                                url_obj = item.get("image_url", {})
                                url = (
                                    url_obj.get("url", "")
                                    if isinstance(url_obj, dict)
                                    else str(url_obj)
                                )
                                if url.startswith("data:"):
                                    try:
                                        header, b64 = url.split(",", 1)
                                        mime = header.split(";")[0].replace("data:", "")
                                        parts.append(
                                            CanonicalContentPart(
                                                type="image", media_type=mime, data=b64, url=url
                                            )
                                        )
                                    except Exception:
                                        parts.append(CanonicalContentPart(type="image", url=url))
                                else:
                                    parts.append(CanonicalContentPart(type="image", url=url))
                    text_content = "\n".join(text_fragments)
                req.messages.append(
                    CanonicalMessage(
                        role="user",
                        content=text_content,
                        parts=parts,
                    )
                )
            elif role == "assistant":
                tc_list: list[CanonicalToolCall] = []
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {"raw": raw_args}
                    tc_list.append(
                        CanonicalToolCall(
                            id=tc.get("id", f"call_{fn_name}"),
                            name=fn_name,
                            arguments=args if isinstance(args, dict) else {"input": args},
                            raw_arguments=raw_args
                            if isinstance(raw_args, str)
                            else json.dumps(raw_args),
                        )
                    )
                req.messages.append(
                    CanonicalMessage(
                        role="assistant",
                        content=content or "",
                        tool_calls=tc_list,
                        thinking=thinking,
                    )
                )
            elif role == "tool":
                req.messages.append(
                    CanonicalMessage(
                        role="tool",
                        tool_results=[
                            CanonicalToolResult(
                                tool_use_id=msg.get("tool_call_id", ""),
                                content=content
                                if isinstance(content, str)
                                else json.dumps(content),
                                name=msg.get("name", ""),
                            )
                        ],
                    )
                )

        return req

    def render_request(self, req: CanonicalRequest) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": req.model,
            "messages": [],
            "stream": req.stream,
        }
        if req.temperature is not None:
            out["temperature"] = req.temperature
        if req.top_p is not None:
            out["top_p"] = req.top_p
        if req.max_tokens is not None:
            out["max_tokens"] = req.max_tokens
        if req.stop_sequences:
            out["stop"] = req.stop_sequences

        if req.system:
            out["messages"].append({"role": "system", "content": req.system})

        for msg in req.messages:
            if msg.role == "user":
                if msg.parts and any(p.type == "image" for p in msg.parts):
                    content_list: list[dict[str, Any]] = []
                    for p in msg.parts:
                        if p.type == "text" and p.text:
                            content_list.append({"type": "text", "text": p.text})
                        elif p.type == "image":
                            if p.url:
                                content_list.append(
                                    {"type": "image_url", "image_url": {"url": p.url}}
                                )
                            elif p.data and p.media_type:
                                content_list.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{p.media_type};base64,{p.data}"
                                        },
                                    }
                                )
                    out["messages"].append({"role": "user", "content": content_list or msg.content})
                else:
                    out["messages"].append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                m: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
                if msg.thinking:
                    m["reasoning_content"] = msg.thinking
                if msg.tool_calls:
                    m["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.raw_arguments or json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                out["messages"].append(m)
            elif msg.role == "tool":
                for tr in msg.tool_results:
                    out["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.tool_use_id,
                            "content": tr.content,
                            "name": tr.name,
                        }
                    )

        if req.tools:
            out["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in req.tools
            ]

        return out

    def parse_response(
        self, payload: dict[str, Any], model: str | None = None
    ) -> CanonicalResponse:
        if "error" in payload:
            return CanonicalResponse(
                content=None,
                model=model or payload.get("model", "default-model"),
                error=payload["error"]
                if isinstance(payload["error"], dict)
                else {"message": str(payload["error"])},
                raw_response=payload,
            )
        choices = payload.get("choices", [])
        if not choices:
            return CanonicalResponse(
                content="", model=model or payload.get("model", "default-model")
            )

        choice = choices[0]
        msg = choice.get("message", {})
        content = msg.get("content")
        thinking = msg.get("reasoning_content") or msg.get("thinking")
        tc_list: list[CanonicalToolCall] = []

        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {"raw": raw_args}
            tc_list.append(
                CanonicalToolCall(
                    id=tc.get("id", f"call_{name}"),
                    name=name,
                    arguments=args if isinstance(args, dict) else {"input": args},
                    raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
                )
            )

        finish_raw = choice.get("finish_reason", "stop")
        finish_map: dict[str, Literal["stop", "length", "tool_calls", "content_filter"]] = {
            "stop": "stop",
            "length": "length",
            "tool_calls": "tool_calls",
            "function_call": "tool_calls",
            "content_filter": "content_filter",
        }
        finish_reason = finish_map.get(finish_raw, "stop")
        if tc_list and finish_reason == "stop":
            finish_reason = "tool_calls"

        usage = payload.get("usage", {})
        return CanonicalResponse(
            content=content,
            model=model or payload.get("model", "default-model"),
            tool_calls=tc_list,
            thinking=thinking,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            raw_response=payload,
        )

    def render_response(self, resp: CanonicalResponse) -> dict[str, Any]:
        if resp.error is not None:
            return {"error": resp.error}
        msg: dict[str, Any] = {"role": "assistant", "content": resp.content}
        if resp.thinking:
            msg["reasoning_content"] = resp.thinking
        if resp.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.raw_arguments or json.dumps(tc.arguments),
                    },
                }
                for tc in resp.tool_calls
            ]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": resp.model,
            "choices": [{"index": 0, "message": msg, "finish_reason": resp.finish_reason}],
            "usage": {
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.prompt_tokens + resp.completion_tokens,
            },
        }

    def parse_chunk(self, chunk: dict[str, Any], state: StreamingState) -> CanonicalChunk:
        choices = chunk.get("choices", [])
        if not choices:
            return CanonicalChunk(raw_chunk=chunk)

        choice = choices[0]
        delta = choice.get("delta", {})
        text_delta = delta.get("content")
        thinking_delta = delta.get("reasoning_content") or delta.get("thinking")

        tool_deltas: list[CanonicalDeltaToolCall] = []
        for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            tc_id = tc.get("id")
            fn = tc.get("function", {})
            fn_name = fn.get("name")
            args_delta = fn.get("arguments", "")
            tool_deltas.append(
                CanonicalDeltaToolCall(
                    index=idx,
                    id=tc_id,
                    name=fn_name,
                    arguments_delta=args_delta,
                )
            )

        finish_raw = choice.get("finish_reason")
        finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None
        if finish_raw:
            f_lower = finish_raw.lower()
            if "tool" in f_lower:
                finish_reason = "tool_calls"
            elif "len" in f_lower or "max" in f_lower:
                finish_reason = "length"
            else:
                finish_reason = "stop"

        usage = chunk.get("usage", {})
        return CanonicalChunk(
            text_delta=text_delta,
            thinking_delta=thinking_delta,
            tool_call_deltas=tool_deltas,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            raw_chunk=chunk,
        )

    def render_chunk(self, chunk: CanonicalChunk, state: StreamingState) -> list[dict[str, Any]]:
        delta: dict[str, Any] = {}
        if chunk.text_delta:
            delta["content"] = chunk.text_delta
        if chunk.thinking_delta:
            delta["reasoning_content"] = chunk.thinking_delta
        if chunk.tool_call_deltas:
            delta["tool_calls"] = [
                {
                    "index": tc.index,
                    **({"id": tc.id} if tc.id else {}),
                    "type": "function",
                    "function": {
                        **({"name": tc.name} if tc.name else {}),
                        "arguments": tc.arguments_delta,
                    },
                }
                for tc in chunk.tool_call_deltas
            ]

        out = {
            "id": state.stream_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": state.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": chunk.finish_reason,
                }
            ],
        }
        if chunk.prompt_tokens or chunk.completion_tokens:
            out["usage"] = {
                "prompt_tokens": chunk.prompt_tokens,
                "completion_tokens": chunk.completion_tokens,
                "total_tokens": chunk.prompt_tokens + chunk.completion_tokens,
            }
        return [out]


class AnthropicProtocolAdapter(BaseProtocolAdapter):
    """Adapter for Anthropic Messages API (/v1/messages)."""

    protocol_name = "anthropic"

    def parse_request(self, payload: dict[str, Any]) -> CanonicalRequest:
        req = CanonicalRequest(
            model=payload.get("model", "default-model"),
            stream=payload.get("stream", False),
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
            max_tokens=payload.get("max_tokens"),
        )
        stops = payload.get("stop_sequences")
        if stops:
            req.stop_sequences = list(stops)

        # System
        sys_val = payload.get("system")
        if isinstance(sys_val, str):
            req.system = sys_val
        elif isinstance(sys_val, list):
            req.system = "\n\n".join(
                b.get("text", "") for b in sys_val if isinstance(b, dict) and b.get("text")
            )

        # Thinking
        th = payload.get("thinking")
        if isinstance(th, dict) and th.get("type") == "enabled":
            req.thinking_budget = th.get("budget_tokens", 2048)

        # Tools
        for t in payload.get("tools", []):
            req.tools.append(
                CanonicalTool(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    parameters=t.get("input_schema", {}),
                )
            )

        # Messages
        for msg in payload.get("messages", []):
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                if isinstance(content, str):
                    req.messages.append(CanonicalMessage(role="user", content=content))
                elif isinstance(content, list):
                    text_parts: list[str] = []
                    parts: list[CanonicalContentPart] = []
                    results: list[CanonicalToolResult] = []
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        b_type = b.get("type")
                        if b_type == "text":
                            t_val = b.get("text", "")
                            text_parts.append(t_val)
                            parts.append(CanonicalContentPart(type="text", text=t_val))
                        elif b_type == "image":
                            source = b.get("source", {})
                            if isinstance(source, dict) and source.get("type") == "base64":
                                parts.append(
                                    CanonicalContentPart(
                                        type="image",
                                        media_type=source.get("media_type", "image/png"),
                                        data=source.get("data", ""),
                                    )
                                )
                        elif b_type == "tool_result":
                            res_content = b.get("content", "")
                            res_text = (
                                res_content
                                if isinstance(res_content, str)
                                else json.dumps(res_content)
                            )
                            results.append(
                                CanonicalToolResult(
                                    tool_use_id=b.get("tool_use_id", ""),
                                    content=res_text,
                                    is_error=b.get("is_error", False),
                                )
                            )
                    if text_parts or parts:
                        req.messages.append(
                            CanonicalMessage(
                                role="user",
                                content="\n".join(text_parts),
                                parts=parts,
                            )
                        )
                    if results:
                        req.messages.append(CanonicalMessage(role="tool", tool_results=results))
            elif role == "assistant":
                if isinstance(content, str):
                    req.messages.append(CanonicalMessage(role="assistant", content=content))
                elif isinstance(content, list):
                    text_parts = []
                    tool_calls = []
                    thinking_text = None
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        b_type = b.get("type")
                        if b_type == "text":
                            text_parts.append(b.get("text", ""))
                        elif b_type == "thinking":
                            thinking_text = b.get("thinking", "")
                        elif b_type == "tool_use":
                            input_val = b.get("input", {})
                            tool_calls.append(
                                CanonicalToolCall(
                                    id=b.get("id", ""),
                                    name=b.get("name", ""),
                                    arguments=input_val
                                    if isinstance(input_val, dict)
                                    else {"input": input_val},
                                    raw_arguments=json.dumps(input_val),
                                )
                            )
                    req.messages.append(
                        CanonicalMessage(
                            role="assistant",
                            content="\n".join(text_parts),
                            tool_calls=tool_calls,
                            thinking=thinking_text,
                        )
                    )

        return req

    def render_request(self, req: CanonicalRequest) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": req.model,
            "messages": [],
            "stream": req.stream,
            "max_tokens": req.max_tokens or 4096,
        }
        if req.system:
            out["system"] = req.system
        if req.temperature is not None:
            out["temperature"] = req.temperature
        if req.top_p is not None:
            out["top_p"] = req.top_p
        if req.stop_sequences:
            out["stop_sequences"] = req.stop_sequences
        if req.thinking_budget:
            out["thinking"] = {"type": "enabled", "budget_tokens": req.thinking_budget}

        # Build raw Anthropic turns while enforcing strictly alternating user / assistant roles
        current_role: str | None = None
        current_blocks: list[dict[str, Any]] = []

        def flush_turn() -> None:
            nonlocal current_role, current_blocks
            if current_role and current_blocks:
                if (
                    len(current_blocks) == 1
                    and isinstance(current_blocks[0], dict)
                    and current_blocks[0].get("type") == "text"
                ):
                    out["messages"].append(
                        {"role": current_role, "content": current_blocks[0]["text"]}
                    )
                else:
                    out["messages"].append({"role": current_role, "content": current_blocks})
                current_blocks = []
                current_role = None

        for msg in req.messages:
            if msg.role == "user":
                if current_role != "user":
                    flush_turn()
                    current_role = "user"
                if msg.parts and any(p.type == "image" for p in msg.parts):
                    for p in msg.parts:
                        if p.type == "text" and p.text:
                            current_blocks.append({"type": "text", "text": p.text})
                        elif p.type == "image" and p.data:
                            current_blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": p.media_type or "image/png",
                                        "data": p.data,
                                    },
                                }
                            )
                elif msg.content:
                    current_blocks.append({"type": "text", "text": msg.content})
            elif msg.role == "tool":
                if current_role != "user":
                    flush_turn()
                    current_role = "user"
                for tr in msg.tool_results:
                    current_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.tool_use_id,
                            "content": tr.content,
                            "is_error": tr.is_error,
                        }
                    )
            elif msg.role == "assistant":
                if current_role != "assistant":
                    flush_turn()
                    current_role = "assistant"
                if msg.thinking:
                    current_blocks.append({"type": "thinking", "thinking": msg.thinking})
                if msg.content:
                    current_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    current_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )

        flush_turn()

        if req.tools:
            out["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in req.tools
            ]

        return out

    def parse_response(
        self, payload: dict[str, Any], model: str | None = None
    ) -> CanonicalResponse:
        if payload.get("type") == "error" or "error" in payload:
            err = payload.get("error", payload)
            return CanonicalResponse(
                content=None,
                model=model or payload.get("model", "claude-3-5-sonnet"),
                error=err if isinstance(err, dict) else {"message": str(err)},
                raw_response=payload,
            )
        content_blocks = payload.get("content", [])
        text_parts: list[str] = []
        thinking_text: str | None = None
        tool_calls: list[CanonicalToolCall] = []

        for b in content_blocks:
            if not isinstance(b, dict):
                continue
            b_type = b.get("type")
            if b_type == "text":
                text_parts.append(b.get("text", ""))
            elif b_type == "thinking":
                thinking_text = b.get("thinking", "")
            elif b_type == "tool_use":
                inp = b.get("input", {})
                tool_calls.append(
                    CanonicalToolCall(
                        id=b.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                        name=b.get("name", ""),
                        arguments=inp if isinstance(inp, dict) else {"input": inp},
                        raw_arguments=json.dumps(inp),
                    )
                )

        stop_raw = payload.get("stop_reason")
        finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"
        if stop_raw == "tool_use" or tool_calls:
            finish_reason = "tool_calls"
        elif stop_raw == "max_tokens":
            finish_reason = "length"

        usage = payload.get("usage", {})
        return CanonicalResponse(
            content="".join(text_parts) if text_parts else None,
            model=model or payload.get("model", "claude-3-5-sonnet"),
            tool_calls=tool_calls,
            thinking=thinking_text,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            raw_response=payload,
        )

    def render_response(self, resp: CanonicalResponse) -> dict[str, Any]:
        if resp.error is not None:
            return {"type": "error", "error": resp.error}
        blocks: list[dict[str, Any]] = []
        if resp.thinking:
            blocks.append({"type": "thinking", "thinking": resp.thinking})
        if resp.content:
            blocks.append({"type": "text", "text": resp.content})
        for tc in resp.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )

        stop_reason = "end_turn"
        if resp.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif resp.finish_reason == "length":
            stop_reason = "max_tokens"

        return {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": resp.model,
            "content": blocks,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": resp.prompt_tokens,
                "output_tokens": resp.completion_tokens,
            },
        }

    def parse_chunk(self, chunk: dict[str, Any], state: StreamingState) -> CanonicalChunk:
        event_type = chunk.get("type")
        text_delta: str | None = None
        thinking_delta: str | None = None
        tool_deltas: list[CanonicalDeltaToolCall] = []
        finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None

        if event_type == "content_block_start":
            cb = chunk.get("content_block", {})
            cb_type = cb.get("type")
            if cb_type == "tool_use":
                tool_deltas.append(
                    CanonicalDeltaToolCall(
                        index=chunk.get("index", 0),
                        id=cb.get("id"),
                        name=cb.get("name"),
                        arguments_delta="",
                    )
                )
        elif event_type == "content_block_delta":
            delta = chunk.get("delta", {})
            d_type = delta.get("type")
            if d_type == "text_delta":
                text_delta = delta.get("text")
            elif d_type == "thinking_delta":
                thinking_delta = delta.get("thinking")
            elif d_type == "input_json_delta":
                tool_deltas.append(
                    CanonicalDeltaToolCall(
                        index=chunk.get("index", 0),
                        arguments_delta=delta.get("partial_json", ""),
                    )
                )
        elif event_type == "message_delta":
            delta = chunk.get("delta", {})
            stop = delta.get("stop_reason")
            if stop == "tool_use":
                finish_reason = "tool_calls"
            elif stop == "max_tokens":
                finish_reason = "length"
            elif stop:
                finish_reason = "stop"

        usage = chunk.get("usage", {})
        return CanonicalChunk(
            text_delta=text_delta,
            thinking_delta=thinking_delta,
            tool_call_deltas=tool_deltas,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            raw_chunk=chunk,
        )

    def render_chunk(self, chunk: CanonicalChunk, state: StreamingState) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        # 1. Emit message_start if first event
        if not state.sent_message_start:
            events.append(
                {
                    "type": "message_start",
                    "message": {
                        "id": state.stream_id,
                        "type": "message",
                        "role": "assistant",
                        "model": state.model,
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": chunk.prompt_tokens, "output_tokens": 0},
                    },
                }
            )
            state.sent_message_start = True

        # 2. Thinking delta
        if chunk.thinking_delta:
            if state.active_block_type != "thinking":
                if state.active_block_type is not None:
                    events.append({"type": "content_block_stop", "index": state.active_block_index})
                    state.current_block_index += 1
                state.active_block_type = "thinking"
                state.active_block_index = state.current_block_index
                events.append(
                    {
                        "type": "content_block_start",
                        "index": state.active_block_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }
                )
            events.append(
                {
                    "type": "content_block_delta",
                    "index": state.active_block_index,
                    "delta": {"type": "thinking_delta", "thinking": chunk.thinking_delta},
                }
            )

        # 3. Text delta
        if chunk.text_delta:
            if state.active_block_type != "text":
                if state.active_block_type is not None:
                    events.append({"type": "content_block_stop", "index": state.active_block_index})
                    state.current_block_index += 1
                state.active_block_type = "text"
                state.active_block_index = state.current_block_index
                events.append(
                    {
                        "type": "content_block_start",
                        "index": state.active_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            events.append(
                {
                    "type": "content_block_delta",
                    "index": state.active_block_index,
                    "delta": {"type": "text_delta", "text": chunk.text_delta},
                }
            )

        # 4. Tool call deltas
        for tc in chunk.tool_call_deltas:
            # New tool call start
            if tc.id or tc.name:
                if state.active_block_type is not None:
                    events.append({"type": "content_block_stop", "index": state.active_block_index})
                    state.current_block_index += 1
                state.active_block_type = "tool_use"
                state.active_block_index = state.current_block_index
                events.append(
                    {
                        "type": "content_block_start",
                        "index": state.active_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.id or f"toolu_{uuid.uuid4().hex[:8]}",
                            "name": tc.name or "tool",
                            "input": {},
                        },
                    }
                )
            if tc.arguments_delta:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": state.active_block_index,
                        "delta": {"type": "input_json_delta", "partial_json": tc.arguments_delta},
                    }
                )

        # 5. Finish events
        if chunk.finish_reason:
            if state.active_block_type is not None:
                events.append({"type": "content_block_stop", "index": state.active_block_index})
                state.active_block_type = None

            stop_reason = "end_turn"
            if chunk.finish_reason == "tool_calls":
                stop_reason = "tool_use"
            elif chunk.finish_reason == "length":
                stop_reason = "max_tokens"

            events.append(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason},
                    "usage": {"output_tokens": chunk.completion_tokens},
                }
            )
            events.append({"type": "message_stop"})

        return events


class GeminiProtocolAdapter(BaseProtocolAdapter):
    """Adapter for Google Gemini API (/v1beta/models/...:generateContent)."""

    protocol_name = "gemini"

    def parse_request(self, payload: dict[str, Any]) -> CanonicalRequest:
        req = CanonicalRequest(model=payload.get("model", "gemini-2.5-pro"))
        gc = payload.get("generationConfig", {})
        req.temperature = gc.get("temperature")
        req.top_p = gc.get("topP")
        req.max_tokens = gc.get("maxOutputTokens")
        if gc.get("stopSequences"):
            req.stop_sequences = list(gc["stopSequences"])

        # System instruction
        si = payload.get("system_instruction", {})
        si_parts = si.get("parts", [])
        if si_parts:
            req.system = "\n\n".join(p.get("text", "") for p in si_parts if "text" in p)

        # Tools
        for t in payload.get("tools", []):
            for fn in t.get("function_declarations", []):
                req.tools.append(
                    CanonicalTool(
                        name=fn.get("name", ""),
                        description=fn.get("description", ""),
                        parameters=fn.get("parameters", {}),
                    )
                )

        # Contents
        for item in payload.get("contents", []):
            role = item.get("role")
            parts = item.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if "text" in p and not p.get("thought")]
            thinking_parts = [p.get("text", "") for p in parts if p.get("thought")]
            combined_text = "\n".join(text_parts)
            combined_thinking = "\n".join(thinking_parts) if thinking_parts else None

            # Multimodal inline data
            canonical_parts: list[CanonicalContentPart] = []
            for p in parts:
                if "text" in p and not p.get("thought"):
                    canonical_parts.append(CanonicalContentPart(type="text", text=p["text"]))
                elif "inline_data" in p:
                    id_obj = p["inline_data"]
                    canonical_parts.append(
                        CanonicalContentPart(
                            type="image",
                            media_type=id_obj.get("mime_type", "image/png"),
                            data=id_obj.get("data", ""),
                        )
                    )

            if role == "user":
                func_responses = [
                    p.get("functionResponse") for p in parts if "functionResponse" in p
                ]
                if func_responses:
                    results = []
                    for fr in func_responses:
                        results.append(
                            CanonicalToolResult(
                                tool_use_id=fr.get("name", ""),
                                content=json.dumps(fr.get("response", {})),
                                name=fr.get("name", ""),
                            )
                        )
                    req.messages.append(CanonicalMessage(role="tool", tool_results=results))
                elif canonical_parts or combined_text:
                    req.messages.append(
                        CanonicalMessage(
                            role="user",
                            content=combined_text,
                            parts=canonical_parts,
                        )
                    )
            elif role == "model":
                func_calls = [p.get("functionCall") for p in parts if "functionCall" in p]
                tc_list = []
                for idx, fc in enumerate(func_calls):
                    tc_list.append(
                        CanonicalToolCall(
                            id=f"call_{fc.get('name')}_{idx}",
                            name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                            raw_arguments=json.dumps(fc.get("args", {})),
                        )
                    )
                req.messages.append(
                    CanonicalMessage(
                        role="assistant",
                        content=combined_text,
                        tool_calls=tc_list,
                        thinking=combined_thinking,
                    )
                )

        return req

    def render_request(self, req: CanonicalRequest) -> dict[str, Any]:
        out: dict[str, Any] = {"contents": []}
        gc: dict[str, Any] = {}
        if req.temperature is not None:
            gc["temperature"] = req.temperature
        if req.top_p is not None:
            gc["topP"] = req.top_p
        if req.max_tokens is not None:
            gc["maxOutputTokens"] = req.max_tokens
        if req.stop_sequences:
            gc["stopSequences"] = req.stop_sequences
        if gc:
            out["generationConfig"] = gc

        if req.system:
            out["system_instruction"] = {"parts": [{"text": req.system}]}

        # Track tool_use_id -> tool_name mapping across request turns so Gemini functionResponse always has valid name
        id_to_name_map: dict[str, str] = {}
        for m in req.messages:
            for tc in m.tool_calls:
                if tc.id and tc.name:
                    id_to_name_map[tc.id] = tc.name

        current_role: str | None = None
        current_parts: list[dict[str, Any]] = []

        def flush_gemini_turn() -> None:
            nonlocal current_role, current_parts
            if current_role and current_parts:
                out["contents"].append({"role": current_role, "parts": current_parts})
                current_parts = []
                current_role = None

        for msg in req.messages:
            if msg.role == "user":
                if current_role != "user":
                    flush_gemini_turn()
                    current_role = "user"
                if msg.parts and any(p.type == "image" for p in msg.parts):
                    for p in msg.parts:
                        if p.type == "text" and p.text:
                            current_parts.append({"text": p.text})
                        elif p.type == "image" and p.data:
                            current_parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": p.media_type or "image/png",
                                        "data": p.data,
                                    }
                                }
                            )
                elif msg.content:
                    current_parts.append({"text": msg.content})
            elif msg.role == "tool":
                if current_role != "user":
                    flush_gemini_turn()
                    current_role = "user"
                for tr in msg.tool_results:
                    try:
                        res_obj = json.loads(tr.content)
                    except Exception:
                        res_obj = {"content": tr.content}
                    # Resolve function name: prefer tr.name, then lookup from prior tool calls, then fallback
                    fn_name = tr.name or id_to_name_map.get(tr.tool_use_id) or tr.tool_use_id
                    current_parts.append(
                        {
                            "functionResponse": {
                                "name": fn_name,
                                "response": res_obj
                                if isinstance(res_obj, dict)
                                else {"text": str(res_obj)},
                            }
                        }
                    )
            elif msg.role == "assistant":
                if current_role != "model":
                    flush_gemini_turn()
                    current_role = "model"
                if msg.thinking:
                    current_parts.append({"thought": True, "text": msg.thinking})
                if msg.content:
                    current_parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    current_parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})

        flush_gemini_turn()

        if req.tools:
            out["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": sanitize_tool_schema_for_gemini(t.parameters),
                        }
                        for t in req.tools
                    ]
                }
            ]

        return out

    def parse_response(
        self, payload: dict[str, Any], model: str | None = None
    ) -> CanonicalResponse:
        if "error" in payload:
            return CanonicalResponse(
                content=None,
                model=model or "gemini-2.5-pro",
                error=payload["error"]
                if isinstance(payload["error"], dict)
                else {"message": str(payload["error"])},
                raw_response=payload,
            )
        candidates = payload.get("candidates", [])
        if not candidates:
            return CanonicalResponse(content="", model=model or "gemini-2.5-pro")

        cand = candidates[0]
        content_obj = cand.get("content", {})
        parts = content_obj.get("parts", [])

        text_parts = [p.get("text", "") for p in parts if "text" in p and not p.get("thought")]
        thinking_parts = [p.get("text", "") for p in parts if p.get("thought")]
        tool_calls: list[CanonicalToolCall] = []

        for idx, p in enumerate(parts):
            if "functionCall" in p:
                fc = p["functionCall"]
                name = fc.get("name", "")
                args = fc.get("args", {})
                tool_calls.append(
                    CanonicalToolCall(
                        id=f"call_{name}_{idx}_{int(time.time())}",
                        name=name,
                        arguments=args if isinstance(args, dict) else {"input": args},
                        raw_arguments=json.dumps(args),
                    )
                )

        finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"
        cf = cand.get("finishReason", "")
        if cf == "MAX_TOKENS":
            finish_reason = "length"
        elif tool_calls or cf in ("TOOL_CALLS", "FUNCTION_CALL"):
            finish_reason = "tool_calls"

        meta = payload.get("usageMetadata", {})
        return CanonicalResponse(
            content="".join(text_parts) if text_parts else None,
            model=model or "gemini-2.5-pro",
            tool_calls=tool_calls,
            thinking="\n".join(thinking_parts) if thinking_parts else None,
            finish_reason=finish_reason,
            prompt_tokens=meta.get("promptTokenCount", 0),
            completion_tokens=meta.get("candidatesTokenCount", 0),
            raw_response=payload,
        )

    def render_response(self, resp: CanonicalResponse) -> dict[str, Any]:
        if resp.error is not None:
            return {"error": resp.error}
        parts: list[dict[str, Any]] = []
        if resp.thinking:
            parts.append({"thought": True, "text": resp.thinking})
        if resp.content:
            parts.append({"text": resp.content})
        for tc in resp.tool_calls:
            parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})

        finish_reason = "STOP"
        if resp.finish_reason == "length":
            finish_reason = "MAX_TOKENS"
        elif resp.finish_reason == "tool_calls":
            finish_reason = "STOP"

        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": parts},
                    "finishReason": finish_reason,
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": resp.prompt_tokens,
                "candidatesTokenCount": resp.completion_tokens,
                "totalTokenCount": resp.prompt_tokens + resp.completion_tokens,
            },
        }

    def parse_chunk(self, chunk: dict[str, Any], state: StreamingState) -> CanonicalChunk:
        candidates = chunk.get("candidates", [])
        if not candidates:
            return CanonicalChunk(raw_chunk=chunk)

        cand = candidates[0]
        content_obj = cand.get("content", {})
        parts = content_obj.get("parts", [])

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_deltas: list[CanonicalDeltaToolCall] = []

        for p in parts:
            if "text" in p:
                if p.get("thought"):
                    thinking_parts.append(p["text"])
                else:
                    text_parts.append(p["text"])
            elif "functionCall" in p:
                fc = p["functionCall"]
                name = fc.get("name", "")
                args = fc.get("args", {})
                args_json = json.dumps(args)
                idx = state.tool_call_indices.get(name, state.next_tool_index)
                if name not in state.tool_call_indices:
                    state.tool_call_indices[name] = idx
                    state.next_tool_index += 1
                tool_deltas.append(
                    CanonicalDeltaToolCall(
                        index=idx,
                        id=f"call_{name}_{idx}",
                        name=name,
                        arguments_delta=args_json,
                    )
                )

        finish_raw = cand.get("finishReason")
        finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None
        if finish_raw:
            if finish_raw == "MAX_TOKENS":
                finish_reason = "length"
            elif tool_deltas or finish_raw in ("TOOL_CALLS", "FUNCTION_CALL"):
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop"

        meta = chunk.get("usageMetadata", {})
        return CanonicalChunk(
            text_delta="".join(text_parts) if text_parts else None,
            thinking_delta="".join(thinking_parts) if thinking_parts else None,
            tool_call_deltas=tool_deltas,
            finish_reason=finish_reason,
            prompt_tokens=meta.get("promptTokenCount", 0),
            completion_tokens=meta.get("candidatesTokenCount", 0),
            raw_chunk=chunk,
        )

    def render_chunk(self, chunk: CanonicalChunk, state: StreamingState) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if chunk.thinking_delta:
            parts.append({"thought": True, "text": chunk.thinking_delta})
        if chunk.text_delta:
            parts.append({"text": chunk.text_delta})
        for tc in chunk.tool_call_deltas:
            try:
                args = json.loads(tc.arguments_delta) if tc.arguments_delta else {}
            except Exception:
                args = {"raw": tc.arguments_delta}
            parts.append({"functionCall": {"name": tc.name or "tool", "args": args}})

        finish_str = None
        if chunk.finish_reason == "length":
            finish_str = "MAX_TOKENS"
        elif chunk.finish_reason:
            finish_str = "STOP"

        out: dict[str, Any] = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": parts},
                    "finishReason": finish_str,
                    "index": 0,
                }
            ],
        }
        if chunk.prompt_tokens or chunk.completion_tokens:
            out["usageMetadata"] = {
                "promptTokenCount": chunk.prompt_tokens,
                "candidatesTokenCount": chunk.completion_tokens,
                "totalTokenCount": chunk.prompt_tokens + chunk.completion_tokens,
            }
        return [out]


# ============================================================================
# 4. PROTOCOL REGISTRY & STREAMING TRANSFORMER
# ============================================================================

# Default provider & ecosystem alias presets mapping to canonical wire protocols
DEFAULT_PROVIDER_ALIASES: dict[str, str] = {
    # Tier 2: Canonical Synonyms
    "claude": "anthropic",
    "claude_code": "anthropic",
    "google": "gemini",
    "antigravity": "gemini",
    "code_assist": "gemini",
    "vertex": "gemini",
    "vertex_gemini": "gemini",
    # Tier 3: OpenAI Wire Protocol Ecosystem Presets
    "copilot": "openai",
    "github_copilot": "openai",
    "microsoft": "openai",
    "azure": "openai",
    "azure_openai": "openai",
    "cursor": "openai",
    "opencode": "openai",
    "openhands": "openai",
    "deepseek": "openai",
    "mistral": "openai",
    "groq": "openai",
    "together": "openai",
    "vllm": "openai",
    "sglang": "openai",
    "ollama": "openai",
    # Tier 3: Anthropic Wire Protocol Ecosystem Presets
    "bedrock": "anthropic",
}


class ProtocolRegistry:
    """Registry maintaining active protocol adapters. Open for 3rd-party extension."""

    _adapters: ClassVar[dict[str, BaseProtocolAdapter]] = {}
    _aliases: ClassVar[dict[str, str]] = dict(DEFAULT_PROVIDER_ALIASES)

    @classmethod
    def register(cls, protocol_name: str, adapter: BaseProtocolAdapter) -> None:
        """Register a protocol adapter instance."""
        cls._adapters[protocol_name.lower().strip()] = adapter

    @classmethod
    def register_alias(cls, alias: str, target_protocol: str) -> None:
        """Register an alias pointing to an existing protocol name."""
        cls._aliases[alias.lower().strip()] = target_protocol.lower().strip()

    @classmethod
    def reset_aliases(cls) -> None:
        """Reset provider aliases to default presets."""
        cls._aliases = dict(DEFAULT_PROVIDER_ALIASES)

    @classmethod
    def resolve_protocol(cls, name: str) -> str:
        """Resolve a provider or alias name to its canonical wire protocol."""
        clean = name.lower().strip()
        return cls._aliases.get(clean, clean)

    @classmethod
    def get(cls, protocol_name: str) -> BaseProtocolAdapter:
        """Retrieve an adapter by protocol or provider alias name."""
        resolved = cls.resolve_protocol(protocol_name)
        if resolved not in cls._adapters:
            protocols_str = ", ".join(repr(p) for p in cls.list_protocols())
            raise ValueError(
                f"Unknown protocol or provider: '{protocol_name}' (resolved to '{resolved}').\n"
                f"Supported canonical wire protocols: {protocols_str}.\n"
                f"To route this provider, either use a canonical wire protocol or register an alias:\n"
                f"    from agent_tool_parser import ProtocolRegistry\n"
                f"    ProtocolRegistry.register_alias({protocol_name!r}, 'openai')  # or 'anthropic' / 'gemini'"
            )
        return cls._adapters[resolved]

    @classmethod
    def list_protocols(cls) -> list[str]:
        """List all currently registered canonical protocol names."""
        return list(cls._adapters.keys())

    @classmethod
    def list_aliases(cls) -> dict[str, str]:
        """List all registered provider aliases and their target protocols."""
        return dict(cls._aliases)


# Default built-in registrations
ProtocolRegistry.register("openai", OpenAIProtocolAdapter())
ProtocolRegistry.register("anthropic", AnthropicProtocolAdapter())
ProtocolRegistry.register("gemini", GeminiProtocolAdapter())


class StreamingProtocolTransformer:
    """Stateful stream transformer converting SSE chunks across protocols in real-time."""

    def __init__(
        self,
        from_format: SupportedProtocol | str,
        to_format: SupportedProtocol | str,
        model: str = "default-model",
        stream_id: str | None = None,
    ):
        self.source_adapter = ProtocolRegistry.get(from_format)
        self.target_adapter = ProtocolRegistry.get(to_format)
        self.state = StreamingState(
            stream_id=stream_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=model,
            source_protocol=from_format,
            target_protocol=to_format,
        )

    def transform_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """Transform a single parsed JSON chunk from source protocol to target protocol events."""
        canonical_chunk = self.source_adapter.parse_chunk(chunk, self.state)
        return self.target_adapter.render_chunk(canonical_chunk, self.state)

    def transform_sse_line(self, line: str) -> list[str]:
        """Convenience method to translate a raw SSE line across wire formats."""
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            return [line]
        if stripped == "data: [DONE]":
            if self.target_adapter.protocol_name == "openai":
                return ["data: [DONE]\n\n"]
            return []

        if stripped.startswith("data: "):
            json_str = stripped[6:].strip()
            try:
                chunk_dict = json.loads(json_str)
            except Exception:
                return [line]
            out_chunks = self.transform_chunk(chunk_dict)
            lines = []
            for oc in out_chunks:
                if self.target_adapter.protocol_name == "anthropic" and "type" in oc:
                    lines.append(f"event: {oc['type']}\ndata: {json.dumps(oc)}\n\n")
                else:
                    lines.append(f"data: {json.dumps(oc)}\n\n")
            return lines
        return [line]


# ============================================================================
# 5. HIGH-LEVEL PUBLIC CONVENIENCE FACADES
# ============================================================================


def translate_request(
    payload: dict[str, Any],
    from_format: SupportedProtocol | str,
    to_format: SupportedProtocol | str,
    override_model: str | None = None,
) -> dict[str, Any]:
    """Convert an LLM request payload from any protocol to any protocol.

    Example:
        gemini_payload = translate_request(
            claude_code_payload,
            from_format="anthropic",
            to_format="gemini",
            override_model="gemini-2.5-pro",
        )
    """
    if from_format == to_format and not override_model:
        return payload

    src_adapter = ProtocolRegistry.get(from_format)
    dst_adapter = ProtocolRegistry.get(to_format)

    ir = src_adapter.parse_request(payload)
    if override_model:
        ir.model = override_model

    return dst_adapter.render_request(ir)


def translate_response(
    payload: dict[str, Any],
    from_format: SupportedProtocol | str,
    to_format: SupportedProtocol | str,
    model: str | None = None,
) -> dict[str, Any]:
    """Convert an LLM response payload from any protocol to any protocol.

    Example:
        openai_resp = translate_response(
            gemini_generate_content_resp,
            from_format="gemini",
            to_format="openai",
            model="gpt-4o",
        )
    """
    if from_format == to_format:
        return payload

    src_adapter = ProtocolRegistry.get(from_format)
    dst_adapter = ProtocolRegistry.get(to_format)

    ir = src_adapter.parse_response(payload, model=model)
    return dst_adapter.render_response(ir)


def create_streaming_transformer(
    from_format: SupportedProtocol | str,
    to_format: SupportedProtocol | str,
    model: str = "default-model",
    stream_id: str | None = None,
) -> StreamingProtocolTransformer:
    """Create a stateful streaming transformer for SSE pipeline conversion."""
    return StreamingProtocolTransformer(
        from_format=from_format,
        to_format=to_format,
        model=model,
        stream_id=stream_id,
    )


class UniversalProtocolAdapter:
    """Consolidated facade maintaining backwards-compatible static and class methods."""

    # Core high-level translation facades
    translate_request = staticmethod(translate_request)
    translate_response = staticmethod(translate_response)
    create_streaming_transformer = staticmethod(create_streaming_transformer)

    # 1. Request Ingestion
    @classmethod
    def request_from_openai(cls, payload: dict[str, Any]) -> CanonicalRequest:
        return ProtocolRegistry.get("openai").parse_request(payload)

    @classmethod
    def request_from_anthropic(cls, payload: dict[str, Any]) -> CanonicalRequest:
        return ProtocolRegistry.get("anthropic").parse_request(payload)

    @classmethod
    def request_from_gemini(cls, payload: dict[str, Any]) -> CanonicalRequest:
        return ProtocolRegistry.get("gemini").parse_request(payload)

    # 2. Request Rendering
    @classmethod
    def request_to_openai(cls, req: CanonicalRequest) -> dict[str, Any]:
        return ProtocolRegistry.get("openai").render_request(req)

    @classmethod
    def request_to_anthropic(cls, req: CanonicalRequest) -> dict[str, Any]:
        return ProtocolRegistry.get("anthropic").render_request(req)

    @classmethod
    def request_to_gemini(cls, req: CanonicalRequest) -> dict[str, Any]:
        return ProtocolRegistry.get("gemini").render_request(req)

    # 3. Response Ingestion
    @classmethod
    def response_from_openai(
        cls, payload: dict[str, Any], model: str | None = None
    ) -> CanonicalResponse:
        return ProtocolRegistry.get("openai").parse_response(payload, model=model)

    @classmethod
    def response_from_anthropic(
        cls, payload: dict[str, Any], model: str | None = None
    ) -> CanonicalResponse:
        return ProtocolRegistry.get("anthropic").parse_response(payload, model=model)

    @classmethod
    def response_from_gemini(
        cls, payload: dict[str, Any], model: str | None = None
    ) -> CanonicalResponse:
        return ProtocolRegistry.get("gemini").parse_response(payload, model=model)

    # 4. Response Rendering
    @classmethod
    def response_to_openai(cls, resp: CanonicalResponse) -> dict[str, Any]:
        return ProtocolRegistry.get("openai").render_response(resp)

    @classmethod
    def response_to_anthropic(cls, resp: CanonicalResponse) -> dict[str, Any]:
        return ProtocolRegistry.get("anthropic").render_response(resp)

    @classmethod
    def response_to_gemini(cls, resp: CanonicalResponse) -> dict[str, Any]:
        return ProtocolRegistry.get("gemini").render_response(resp)


# ============================================================================
# 6. TOOL & SCHEMA INTEROPERABILITY (MCP, OPENAI, ANTHROPIC, GEMINI, A2A)
# ============================================================================


@dataclass
class ToolMetadata:
    """Execution context and governance metadata for ToolCalls in A2A and MCP environments."""

    caller_agent: str | None = None
    target_agent: str | None = None
    task_id: str | None = None
    parent_call_id: str | None = None
    category: Any = None
    is_mutation: bool = False
    is_verification: bool = False
    requires_approval: bool = False
    idempotent: bool = False
    security_level: str = "standard"
    owner_agent: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
    custom_annotations: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.owner_agent and not self.caller_agent:
            self.caller_agent = self.owner_agent
        elif self.caller_agent and not self.owner_agent:
            self.owner_agent = self.caller_agent

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary representation."""
        res: dict[str, Any] = {
            "caller_agent": self.caller_agent,
            "owner_agent": self.owner_agent,
            "target_agent": self.target_agent,
            "task_id": self.task_id,
            "parent_call_id": self.parent_call_id,
            "category": str(self.category) if self.category is not None else None,
            "is_mutation": self.is_mutation,
            "is_verification": self.is_verification,
            "requires_approval": self.requires_approval,
            "idempotent": self.idempotent,
            "security_level": self.security_level,
            "artifact_ids": list(self.artifact_ids),
            "custom_annotations": dict(self.custom_annotations),
            "annotations": dict(self.custom_annotations),
        }
        if self.extra:
            res["extra"] = dict(self.extra)
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolMetadata:
        """Construct ToolMetadata from dictionary representation."""
        if not isinstance(data, dict):
            return cls()
        return cls(
            caller_agent=data.get("caller_agent") or data.get("owner_agent"),
            owner_agent=data.get("owner_agent") or data.get("caller_agent"),
            target_agent=data.get("target_agent"),
            task_id=data.get("task_id"),
            parent_call_id=data.get("parent_call_id"),
            category=data.get("category"),
            is_mutation=bool(data.get("is_mutation", False)),
            is_verification=bool(data.get("is_verification", False)),
            requires_approval=bool(data.get("requires_approval", False)),
            idempotent=bool(data.get("idempotent", False)),
            security_level=data.get("security_level", "standard"),
            artifact_ids=list(data.get("artifact_ids") or []),
            custom_annotations=dict(
                data.get("custom_annotations") or data.get("annotations") or {}
            ),
            extra=dict(data.get("extra") or {}),
        )

    def to_headers(self, prefix: str = "x-a2a-") -> dict[str, str]:
        """Convert metadata into standard HTTP / A2A RPC metadata headers."""
        headers: dict[str, str] = {}
        caller = self.caller_agent or self.owner_agent
        if caller:
            headers[f"{prefix}caller-agent"] = caller
        if self.target_agent:
            headers[f"{prefix}target-agent"] = self.target_agent
        if self.task_id:
            headers[f"{prefix}task-id"] = self.task_id
        if self.parent_call_id:
            headers[f"{prefix}parent-call-id"] = self.parent_call_id
        if self.category is not None:
            headers[f"{prefix}category"] = str(self.category)
        if self.is_mutation:
            headers[f"{prefix}is-mutation"] = "true"
        if self.is_verification:
            headers[f"{prefix}is-verification"] = "true"
        if self.requires_approval:
            headers[f"{prefix}requires-approval"] = "true"
        if self.idempotent:
            headers[f"{prefix}idempotent"] = "true"
        if self.security_level != "standard":
            headers[f"{prefix}security-level"] = self.security_level
        if self.artifact_ids:
            headers[f"{prefix}artifact-ids"] = ",".join(self.artifact_ids)
        for k, v in self.custom_annotations.items():
            headers[f"{prefix}meta-{k}"] = str(v)
        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str], prefix: str = "x-a2a-") -> ToolMetadata:
        """Parse ToolMetadata from incoming HTTP / A2A RPC metadata headers."""
        norm = {k.lower(): v for k, v in headers.items()}
        caller = norm.get(f"{prefix}caller-agent")
        target = norm.get(f"{prefix}target-agent")
        task = norm.get(f"{prefix}task-id")
        parent = norm.get(f"{prefix}parent-call-id")
        cat = norm.get(f"{prefix}category")
        is_mut = norm.get(f"{prefix}is-mutation", "").lower() in ("true", "1")
        is_ver = norm.get(f"{prefix}is-verification", "").lower() in ("true", "1")
        req_app = norm.get(f"{prefix}requires-approval", "").lower() in ("true", "1")
        idem = norm.get(f"{prefix}idempotent", "").lower() in ("true", "1")
        sec = norm.get(f"{prefix}security-level", "standard")
        art_raw = norm.get(f"{prefix}artifact-ids", "")
        artifacts = [a.strip() for a in art_raw.split(",") if a.strip()] if art_raw else []

        meta_prefix = f"{prefix}meta-"
        custom: dict[str, Any] = {}
        for k, v in norm.items():
            if k.startswith(meta_prefix):
                custom[k[len(meta_prefix) :]] = v

        return cls(
            caller_agent=caller,
            owner_agent=caller,
            target_agent=target,
            task_id=task,
            parent_call_id=parent,
            category=cat,
            is_mutation=is_mut,
            is_verification=is_ver,
            requires_approval=req_app,
            idempotent=idem,
            security_level=sec,
            artifact_ids=artifacts,
            custom_annotations=custom,
        )


@dataclass
class ToolDefinition:
    """Canonical representation of an LLM tool / function definition."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: ToolMetadata | None = None


@dataclass
class ToolResult:
    """Canonical representation of a tool execution outcome."""

    tool_call_id: str
    name: str = ""
    content: str | list[dict[str, Any]] = ""
    is_error: bool = False
    metadata: ToolMetadata | None = None


def sanitize_tool_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Sanitize and reduce rich JSON Schema (from MCP/OpenAI) for Google Gemini compatibility.

    Google Gemini API strictly rejects schemas containing '$ref', '$defs', 'definitions',
    'anyOf', 'oneOf', 'default', 'examples', and complex union types with HTTP 400.
    This sanitizer resolves references and flattens unions into compliant OpenAPI 3.0 schemas.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    defs: dict[str, Any] = {}
    if "$defs" in schema and isinstance(schema["$defs"], dict):
        defs.update(schema["$defs"])
    if "definitions" in schema and isinstance(schema["definitions"], dict):
        defs.update(schema["definitions"])

    def _resolve_and_clean(node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        # 1. Resolve $ref if present
        if "$ref" in node and isinstance(node["$ref"], str):
            ref_path = node["$ref"]
            ref_key = ref_path.split("/")[-1]
            if ref_key in defs:
                resolved = copy.deepcopy(defs[ref_key])
                for k, v in node.items():
                    if k != "$ref":
                        resolved[k] = v
                return _resolve_and_clean(resolved)

        # 2. Flatten anyOf / oneOf
        union_list = node.get("anyOf") or node.get("oneOf")
        if union_list and isinstance(union_list, list):
            non_null = [b for b in union_list if isinstance(b, dict) and b.get("type") != "null"]
            if non_null:
                primary = copy.deepcopy(non_null[0])
                for k, v in node.items():
                    if k not in ("anyOf", "oneOf"):
                        primary.setdefault(k, v)
                return _resolve_and_clean(primary)
            return {"type": "string"}

        cleaned: dict[str, Any] = {}
        disallowed_keys = {
            "$schema",
            "$id",
            "$defs",
            "definitions",
            "default",
            "examples",
            "patternProperties",
            "additionalProperties",
            "$comment",
        }

        for k, v in node.items():
            if k in disallowed_keys:
                continue
            if k == "properties" and isinstance(v, dict):
                cleaned["properties"] = {
                    prop_name: _resolve_and_clean(prop_val) for prop_name, prop_val in v.items()
                }
            elif k == "items" and isinstance(v, dict):
                cleaned["items"] = _resolve_and_clean(v)
            elif k == "type" and isinstance(v, list):
                types = [t for t in v if t != "null"]
                cleaned["type"] = types[0] if types else "string"
            else:
                cleaned[k] = v

        if "type" not in cleaned and "properties" in cleaned:
            cleaned["type"] = "object"

        return cleaned

    res = _resolve_and_clean(schema)
    return res if isinstance(res, dict) else {}


def sanitize_tool_schema_for_openai(schema: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    """Sanitize JSON Schema for OpenAI function calling.

    If strict=True, enforces OpenAI Structured Outputs constraints:
    - additionalProperties: False on all objects
    - all properties explicitly listed in required array
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    res = copy.deepcopy(schema)
    res.pop("$schema", None)
    res.pop("$id", None)

    if not strict:
        return res

    def _apply_strict(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            node["additionalProperties"] = False
            props = node.get("properties", {})
            if isinstance(props, dict):
                node["required"] = list(props.keys())
                for prop_val in props.values():
                    _apply_strict(prop_val)
        elif node.get("type") == "array" and "items" in node:
            _apply_strict(node["items"])

    _apply_strict(res)
    return res


def convert_tool_definitions(
    tools: list[Any],
    from_format: str = "auto",
    to_format: str = "openai",
) -> list[dict[str, Any]]:
    """Convert tool specifications between OpenAI, Anthropic, Gemini, and MCP schemas.

    Supported formats: 'openai', 'anthropic', 'gemini', 'mcp' (plus registered provider aliases).
    Automatically sanitizes schemas when translating to Gemini to avoid HTTP 400.
    """
    src = ProtocolRegistry.resolve_protocol(from_format) if from_format != "auto" else "auto"
    dst = ProtocolRegistry.resolve_protocol(to_format)

    # 1. Normalize input to list[ToolDefinition]
    definitions: list[ToolDefinition] = []
    for t in tools:
        if isinstance(t, ToolDefinition):
            definitions.append(t)
            continue
        if not isinstance(t, dict):
            continue
        item_src = src
        if item_src == "auto":
            if "inputSchema" in t:
                item_src = "mcp"
            elif "input_schema" in t:
                item_src = "anthropic"
            elif "function_declarations" in t or (
                "parameters" in t and "function" not in t and "type" not in t
            ):
                item_src = "gemini"
            else:
                item_src = "openai"

        if item_src == "mcp":
            definitions.append(
                ToolDefinition(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    parameters=t.get("inputSchema", {}),
                )
            )
        elif item_src == "anthropic":
            definitions.append(
                ToolDefinition(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    parameters=t.get("input_schema", {}),
                )
            )
        elif item_src == "gemini":
            # Can be {"function_declarations": [...]} or direct declaration
            decls = t.get("function_declarations", [t])
            for d in decls:
                definitions.append(
                    ToolDefinition(
                        name=d.get("name", ""),
                        description=d.get("description", ""),
                        parameters=d.get("parameters", {}),
                    )
                )
        else:
            # Default / OpenAI format
            if t.get("type") == "function" and "function" in t:
                fn = t["function"]
                definitions.append(
                    ToolDefinition(
                        name=fn.get("name", ""),
                        description=fn.get("description", ""),
                        parameters=fn.get("parameters", {}),
                    )
                )
            elif "name" in t:
                definitions.append(
                    ToolDefinition(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        parameters=t.get("parameters", {}),
                    )
                )

    # 2. Render to destination format
    if dst == "mcp":
        return [
            {
                "name": td.name,
                "description": td.description,
                "inputSchema": td.parameters,
            }
            for td in definitions
        ]
    elif dst == "anthropic":
        return [
            {
                "name": td.name,
                "description": td.description,
                "input_schema": td.parameters,
            }
            for td in definitions
        ]
    elif dst == "gemini":
        gemini_decls = [
            {
                "name": td.name,
                "description": td.description,
                "parameters": sanitize_tool_schema_for_gemini(td.parameters),
            }
            for td in definitions
        ]
        return [{"function_declarations": gemini_decls}] if gemini_decls else []
    else:
        # Default / OpenAI
        return [
            {
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters,
                },
            }
            for td in definitions
        ]


def _extract_canonical_tool_calls(
    calls: Any,
    from_format: str = "auto",
) -> list[CanonicalToolCall]:
    """Helper to extract a list of CanonicalToolCall objects from various formats."""
    src = ProtocolRegistry.resolve_protocol(from_format) if from_format != "auto" else "auto"
    canonical_calls: list[CanonicalToolCall] = []

    if src in ("raw", "text") or isinstance(calls, str):
        from agent_tool_parser.parser import try_parse_tool_calls

        parsed = try_parse_tool_calls(calls) if isinstance(calls, str) else []
        for idx, tc in enumerate(parsed):
            args = getattr(tc, "args", {})
            canonical_calls.append(
                CanonicalToolCall(
                    id=getattr(tc, "id", None) or f"call_{tc.name}_{idx}",
                    name=tc.name,
                    arguments=args if isinstance(args, dict) else {"input": args},
                    raw_arguments=json.dumps(args),
                )
            )
        return canonical_calls

    items = [calls] if not isinstance(calls, list) else calls
    for idx, item in enumerate(items):
        if isinstance(item, CanonicalToolCall):
            canonical_calls.append(item)
            continue
        if hasattr(item, "name") and hasattr(item, "args"):
            args = getattr(item, "args", {})
            canonical_calls.append(
                CanonicalToolCall(
                    id=getattr(item, "id", None) or f"call_{item.name}_{idx}",
                    name=item.name,
                    arguments=args if isinstance(args, dict) else {"input": args},
                    raw_arguments=json.dumps(args),
                )
            )
            continue
        if not isinstance(item, dict):
            continue

        item_src = src
        if item_src == "auto":
            if "functionCall" in item:
                item_src = "gemini"
            elif item.get("type") == "tool_use":
                item_src = "anthropic"
            elif item.get("method") == "tools/call" or (
                isinstance(item.get("params"), dict) and "arguments" in item["params"]
            ):
                item_src = "mcp"
            elif "function" in item:
                item_src = "openai"
            elif "name" in item:
                args = item.get("arguments", item.get("args", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                canonical_calls.append(
                    CanonicalToolCall(
                        id=item.get("id", f"call_{item.get('name')}_{idx}"),
                        name=item.get("name", ""),
                        arguments=args if isinstance(args, dict) else {"input": args},
                        raw_arguments=json.dumps(args),
                    )
                )
                continue
            else:
                item_src = "openai"

        if item_src == "gemini" or "functionCall" in item:
            fc = item.get("functionCall", item)
            canonical_calls.append(
                CanonicalToolCall(
                    id=f"call_{fc.get('name')}_{idx}",
                    name=fc.get("name", ""),
                    arguments=fc.get("args", {}),
                    raw_arguments=json.dumps(fc.get("args", {})),
                )
            )
        elif item_src == "anthropic" or item.get("type") == "tool_use":
            canonical_calls.append(
                CanonicalToolCall(
                    id=item.get("id", f"toolu_{idx}"),
                    name=item.get("name", ""),
                    arguments=item.get("input", {}),
                    raw_arguments=json.dumps(item.get("input", {})),
                )
            )
        elif item_src == "mcp" or item.get("method") == "tools/call":
            params = item.get("params", item)
            canonical_calls.append(
                CanonicalToolCall(
                    id=f"mcp_call_{idx}",
                    name=params.get("name", ""),
                    arguments=params.get("arguments", {}),
                    raw_arguments=json.dumps(params.get("arguments", {})),
                )
            )
        else:
            # OpenAI format
            fn = item.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {"raw": raw_args}
            canonical_calls.append(
                CanonicalToolCall(
                    id=item.get("id", f"call_{fn.get('name')}_{idx}"),
                    name=fn.get("name", ""),
                    arguments=args if isinstance(args, dict) else {"input": args},
                    raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
                )
            )

    return canonical_calls


def normalize_to_ir(
    item: Any,
    from_format: str = "auto",
) -> CanonicalToolCall | list[CanonicalToolCall]:
    """Normalize tool calls (raw string, ToolCall, dict, or list thereof) into CanonicalToolCall IR.

    Returns a single CanonicalToolCall if passed a single object, or list[CanonicalToolCall]
    if passed a list or raw multi-call string.
    """
    canonical = _extract_canonical_tool_calls(item, from_format=from_format)
    if isinstance(item, list):
        return canonical
    if len(canonical) == 1 and not isinstance(item, str):
        return canonical[0]
    return canonical


def convert_tool_calls(
    calls: Any,
    from_format: str = "auto",
    to_format: str = "openai",
) -> list[dict[str, Any]]:
    """Convert tool call invocations across OpenAI, Anthropic, Gemini, MCP, or raw model text."""
    dst = ProtocolRegistry.resolve_protocol(to_format)
    canonical_calls = _extract_canonical_tool_calls(calls, from_format=from_format)

    # 2. Render
    if dst == "mcp":
        return [
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            }
            for tc in canonical_calls
        ]
    elif dst == "anthropic":
        return [
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            }
            for tc in canonical_calls
        ]
    elif dst == "gemini":
        return [
            {
                "functionCall": {
                    "name": tc.name,
                    "args": tc.arguments,
                }
            }
            for tc in canonical_calls
        ]
    else:
        # Default / OpenAI
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.raw_arguments or json.dumps(tc.arguments),
                },
            }
            for tc in canonical_calls
        ]


def convert_tool_results(
    results: list[dict[str, Any]] | dict[str, Any],
    from_format: str = "auto",
    to_format: str = "openai",
    tool_name_map: dict[str, str] | None = None,
    name_map: dict[str, str] | None = None,
    default_tool_call_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert tool execution results across OpenAI, Anthropic, Gemini, and MCP."""
    src = ProtocolRegistry.resolve_protocol(from_format) if from_format != "auto" else "auto"
    dst = ProtocolRegistry.resolve_protocol(to_format)
    resolved_name_map = tool_name_map or name_map or {}

    results_list: list[dict[str, Any]] = [results] if isinstance(results, dict) else results

    canonical_results: list[ToolResult] = []

    # 1. Ingest
    for idx, r in enumerate(results_list):
        if not isinstance(r, dict):
            continue
        item_src = src
        if item_src == "auto":
            if "functionResponse" in r:
                item_src = "gemini"
            elif r.get("type") == "tool_result":
                item_src = "anthropic"
            elif "isError" in r or (
                isinstance(r.get("content"), list)
                and any(isinstance(c, dict) and "type" in c for c in r.get("content", []))
            ):
                item_src = "mcp"
            else:
                item_src = "openai"

        if item_src == "mcp":
            content_val = r.get("content", [])
            call_id = default_tool_call_id if (default_tool_call_id and idx == 0) else f"call_{idx}"
            canonical_results.append(
                ToolResult(
                    tool_call_id=call_id,
                    content=content_val,
                    is_error=r.get("isError", False),
                )
            )
        elif item_src == "anthropic" or r.get("type") == "tool_result":
            canonical_results.append(
                ToolResult(
                    tool_call_id=r.get("tool_use_id", ""),
                    content=r.get("content", ""),
                    is_error=r.get("is_error", False),
                )
            )
        elif item_src == "gemini" or "functionResponse" in r:
            fr = r.get("functionResponse", r)
            resp_obj = fr.get("response", {})
            content_val = (
                resp_obj.get("content", resp_obj) if isinstance(resp_obj, dict) else resp_obj
            )
            canonical_results.append(
                ToolResult(
                    tool_call_id=fr.get("name", ""),
                    name=fr.get("name", ""),
                    content=content_val,
                )
            )
        else:
            # OpenAI format
            canonical_results.append(
                ToolResult(
                    tool_call_id=r.get("tool_call_id", ""),
                    name=r.get("name", ""),
                    content=r.get("content", ""),
                )
            )

    # 2. Render
    if dst == "mcp":
        mcp_blocks: list[dict[str, Any]] = []
        is_err = False
        for tr in canonical_results:
            if tr.is_error:
                is_err = True
            if isinstance(tr.content, list):
                mcp_blocks.extend(tr.content)
            else:
                mcp_blocks.append({"type": "text", "text": str(tr.content)})
        return [{"content": mcp_blocks, "isError": is_err}]
    elif dst == "anthropic":
        return [
            {
                "type": "tool_result",
                "tool_use_id": tr.tool_call_id,
                "content": tr.content,
                "is_error": tr.is_error,
            }
            for tr in canonical_results
        ]
    elif dst == "gemini":
        parts: list[dict[str, Any]] = []
        for tr in canonical_results:
            fn_name = tr.name or resolved_name_map.get(tr.tool_call_id) or tr.tool_call_id
            resp_payload = (
                tr.content if isinstance(tr.content, (dict, list)) else {"content": str(tr.content)}
            )
            parts.append(
                {
                    "functionResponse": {
                        "name": fn_name,
                        "response": resp_payload,
                    }
                }
            )
        return parts
    else:
        # Default / OpenAI
        return [
            {
                "role": "tool",
                "tool_call_id": tr.tool_call_id,
                "name": tr.name or resolved_name_map.get(tr.tool_call_id, ""),
                "content": json.dumps(tr.content)
                if isinstance(tr.content, (dict, list))
                else str(tr.content),
            }
            for tr in canonical_results
        ]


def to_mcp_tool_call(output: str | dict[str, Any]) -> dict[str, Any]:
    """Extract tool call from raw model output or parsed dictionary and return standard MCP tools/call JSON-RPC."""
    if isinstance(output, str):
        calls = convert_tool_calls(output, from_format="raw", to_format="mcp")
        if calls:
            return calls[0]
        return {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "", "arguments": {}}}
    elif isinstance(output, dict):
        calls = convert_tool_calls([output], from_format="openai", to_format="mcp")
        if calls:
            return calls[0]
        name = output.get("name", "")
        args = output.get("arguments") or output.get("args") or {}
        return {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
    return {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "", "arguments": {}}}


def to_mcp_call_result(content: Any, is_error: bool = False) -> dict[str, Any]:
    """Create a standard MCP CallToolResult from text, dictionary, or content blocks."""
    if isinstance(content, list):
        if content and all(isinstance(item, dict) and "type" in item for item in content):
            return {"content": content, "isError": is_error}
        text_val = json.dumps(content)
        return {"content": [{"type": "text", "text": text_val}], "isError": is_error}
    text_val = json.dumps(content) if isinstance(content, dict) else str(content)
    return {"content": [{"type": "text", "text": text_val}], "isError": is_error}
