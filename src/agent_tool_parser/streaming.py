# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from dataclasses import dataclass

from agent_tool_parser.models import ToolCall
from agent_tool_parser.parser import ToolParser

_ATTR_NAME_RE = re.compile(
    r"\b(?:name|tool|tool_name|function)\s*=\s*['\"]?([a-zA-Z0-9_.:-]+)['\"]?",
    re.IGNORECASE,
)

_INVOKE_OPEN_RE = re.compile(
    r"<[｜|]*(?:dsml[｜|]*)?(?P<tag>tool_invoke|invoke|tool_call|call|tool|invocation|function_call|function|action|tool_use|ant_tool_use|function_use)(?::(?P<colon_tool>[\w-]+))?\b(?P<attrs>[^>]*)>",
    re.IGNORECASE,
)

_TAG_CLOSE_RE = re.compile(
    r"</[｜|]*(?:dsml[｜|]*)?(?:tool_invoke|invoke|tool_call|call|tool|invocation|function_call|function|action|tool_use|ant_tool_use|function_use)(?::[\w-]+)?\s*>",
    re.IGNORECASE,
)

_NATIVE_TOKEN_START_RE = re.compile(
    r"<[｜|]*tool call begin[｜|]*>(?:function=)?(?P<name>[\w.:-]+)(?:<[｜|]*tool sep[｜|]*>)?",
    re.IGNORECASE,
)
_NATIVE_TOKEN_END_RE = re.compile(r"<[｜|]*tool call end[｜|]*>", re.IGNORECASE)

_QWEN_START_RE = re.compile(
    r"<\|action_start\|><\|action_name\|>(?P<name>[\w.:-]+)<\|action_args\|>",
    re.IGNORECASE,
)
_QWEN_END_RE = re.compile(r"<\|action_end\|>", re.IGNORECASE)

_REACT_START_RE = re.compile(
    r"Action:\s*(?P<name>[\w.:-]+)\s*\n(?:Action Input:\s*)?",
    re.IGNORECASE,
)

_CODEBLOCK_START_RE = re.compile(r"```(?:json)?\s*\{", re.IGNORECASE)
_JSON_NAME_RE = re.compile(
    r"['\"](?:name|tool|action|function)['\"]\s*:\s*['\"](?P<name>[\w.:-]+)['\"]",
    re.IGNORECASE,
)


@dataclass
class StreamEvent:
    """Event emitted during token-by-token stream parsing."""

    event_type: str
    text: str | None = None
    thinking: str | None = None
    name: str | None = None
    call_id: str | None = None
    delta: str | None = None
    tool_call: ToolCall | None = None

    def __init__(
        self,
        event_type: str,
        text: str | None = None,
        thinking: str | None = None,
        name: str | None = None,
        call_id: str | None = None,
        delta: str | None = None,
        tool_call: ToolCall | None = None,
        content: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        self.event_type = event_type
        if content is not None:
            if event_type == "thinking":
                thinking = content
            elif event_type == "text":
                text = content
            elif event_type == "tool_call_arguments_chunk":
                delta = content
        if tool_name is not None and name is None:
            name = tool_name
        self.text = text
        self.thinking = thinking
        self.name = name
        self.call_id = call_id
        self.delta = delta
        self.tool_call = tool_call

    @property
    def is_text(self) -> bool:
        return self.event_type == "text"

    @property
    def is_thinking(self) -> bool:
        return self.event_type == "thinking"

    @property
    def is_tool_call_started(self) -> bool:
        return self.event_type == "tool_call_started"

    @property
    def is_tool_call_arguments_chunk(self) -> bool:
        return self.event_type == "tool_call_arguments_chunk"

    @property
    def is_tool_call_completed(self) -> bool:
        return self.event_type == "tool_call_completed"

    @property
    def is_tool_call(self) -> bool:
        return self.event_type in (
            "tool_call_started",
            "tool_call_arguments_chunk",
            "tool_call_completed",
        )

    @property
    def content(self) -> str:
        return self.text or self.thinking or self.delta or ""

    @property
    def tool_name(self) -> str | None:
        if self.name:
            return self.name
        if self.tool_call:
            return self.tool_call.name
        return None


class StreamingToolParser:
    """Token-by-token streaming parser for heterogeneous LLM tool calls.

    Consumes incremental text chunks from streaming endpoints (SSE, async iterators)
    and emits real-time events for text, thinking blocks, early tool starts, argument
    deltas, and completed invocations.
    """

    def __init__(
        self,
        allowed_tools: set[str] | list[str] | None = None,
        tool_aliases: dict[str, str] | None = None,
        param_aliases: dict[str, str] | None = None,
        allow_shell_fallback: bool = False,
    ) -> None:
        self.parser = ToolParser(
            allowed_tools=allowed_tools,
            tool_aliases=tool_aliases,
            param_aliases=param_aliases,
            allow_shell_fallback=allow_shell_fallback,
        )
        self.buffer = ""
        self.state = "normal"
        self._current_name: str | None = None
        self._current_call_id: str | None = None
        self._current_format: str | None = None
        self._last_emitted_body_len = 0

    def reset(self) -> None:
        """Reset internal buffer and streaming state."""
        self.buffer = ""
        self.state = "normal"
        self._current_name = None
        self._current_call_id = None
        self._current_format = None
        self._last_emitted_body_len = 0

    def feed(self, chunk: str) -> list[StreamEvent]:
        """Feed an incremental chunk of tokens into the parser."""
        self.buffer += chunk
        events: list[StreamEvent] = []

        while True:
            if self.state == "normal":
                # 1. Thinking block <think>
                think_idx = self.buffer.find("<think>")
                if think_idx != -1:
                    if think_idx > 0:
                        events.append(StreamEvent(event_type="text", text=self.buffer[:think_idx]))
                    self.buffer = self.buffer[think_idx + 7 :]
                    self.state = "thinking"
                    continue

                # 2. XML / DSML invoke tag
                m = _INVOKE_OPEN_RE.search(self.buffer)
                if m:
                    start = m.start()
                    end = m.end()
                    colon_tool = m.group("colon_tool")
                    attrs = m.group("attrs") or ""

                    raw_name: str | None = None
                    if colon_tool and colon_tool.strip():
                        raw_name = colon_tool.strip()
                    else:
                        attr_m = _ATTR_NAME_RE.search(attrs)
                        if attr_m:
                            raw_name = attr_m.group(1).strip()

                    if raw_name:
                        norm_name = self.parser.normalize_name(raw_name)
                        if start > 0:
                            events.append(StreamEvent(event_type="text", text=self.buffer[:start]))

                        events.append(
                            StreamEvent(
                                event_type="tool_call_started",
                                name=norm_name,
                                call_id=None,
                            )
                        )
                        self.buffer = self.buffer[end:]
                        self.state = "in_tool_call"
                        self._current_name = norm_name
                        self._current_call_id = None
                        self._current_format = "xml_dsml"
                        self._last_emitted_body_len = 0
                        continue

                # 3. Native DeepSeek tokens
                nm = _NATIVE_TOKEN_START_RE.search(self.buffer)
                if nm:
                    start = nm.start()
                    end = nm.end()
                    name = nm.group("name").strip()
                    norm_name = self.parser.normalize_name(name)

                    if start > 0:
                        events.append(StreamEvent(event_type="text", text=self.buffer[:start]))

                    events.append(
                        StreamEvent(
                            event_type="tool_call_started",
                            name=norm_name,
                            call_id=None,
                        )
                    )
                    self.buffer = self.buffer[end:]
                    self.state = "in_tool_call"
                    self._current_name = norm_name
                    self._current_call_id = None
                    self._current_format = "native_token"
                    self._last_emitted_body_len = 0
                    continue

                # 4. Qwen Action
                qm = _QWEN_START_RE.search(self.buffer)
                if qm:
                    start = qm.start()
                    end = qm.end()
                    name = qm.group("name").strip()
                    norm_name = self.parser.normalize_name(name)

                    if start > 0:
                        events.append(StreamEvent(event_type="text", text=self.buffer[:start]))

                    events.append(
                        StreamEvent(
                            event_type="tool_call_started",
                            name=norm_name,
                            call_id=None,
                        )
                    )
                    self.buffer = self.buffer[end:]
                    self.state = "in_tool_call"
                    self._current_name = norm_name
                    self._current_call_id = None
                    self._current_format = "qwen"
                    self._last_emitted_body_len = 0
                    continue

                # 5. ReAct pattern
                rm = _REACT_START_RE.search(self.buffer)
                if rm:
                    start = rm.start()
                    end = rm.end()
                    name = rm.group("name").strip()
                    norm_name = self.parser.normalize_name(name)

                    if start > 0:
                        events.append(StreamEvent(event_type="text", text=self.buffer[:start]))

                    events.append(
                        StreamEvent(
                            event_type="tool_call_started",
                            name=norm_name,
                            call_id=None,
                        )
                    )
                    self.buffer = self.buffer[end:]
                    self.state = "in_tool_call"
                    self._current_name = norm_name
                    self._current_call_id = None
                    self._current_format = "react"
                    self._last_emitted_body_len = 0
                    continue

                # 6. Markdown JSON codeblock
                cb_idx = self.buffer.find("```")
                if cb_idx != -1:
                    cbm = _CODEBLOCK_START_RE.search(self.buffer[cb_idx:])
                    if cbm:
                        after_start = self.buffer[cb_idx + cbm.end() :]
                        jnm = _JSON_NAME_RE.search(after_start)
                        if jnm:
                            name = jnm.group("name").strip()
                            norm_name = self.parser.normalize_name(name)

                            if cb_idx > 0:
                                events.append(
                                    StreamEvent(event_type="text", text=self.buffer[:cb_idx])
                                )

                            events.append(
                                StreamEvent(
                                    event_type="tool_call_started",
                                    name=norm_name,
                                    call_id=None,
                                )
                            )
                            self.buffer = self.buffer[cb_idx:]
                            self.state = "in_tool_call"
                            self._current_name = norm_name
                            self._current_call_id = None
                            self._current_format = "markdown_json"
                            self._last_emitted_body_len = 0
                            continue

                    # If not matched as tool call yet, check if it closed
                    after_fence = self.buffer[cb_idx + 3 :]
                    close_fence = after_fence.find("```")
                    if close_fence != -1:
                        # Non-tool codeblock closed
                        full_code_end = cb_idx + 3 + close_fence + 3
                        events.append(
                            StreamEvent(event_type="text", text=self.buffer[:full_code_end])
                        )
                        self.buffer = self.buffer[full_code_end:]
                        continue

                    # Still open, might become a tool call or close soon.
                    # Emit any text preceding the codeblock
                    if cb_idx > 0:
                        events.append(StreamEvent(event_type="text", text=self.buffer[:cb_idx]))
                        self.buffer = self.buffer[cb_idx:]

                    # Wait for more chunks (unless buffer grows too large without opening a JSON object)
                    if len(self.buffer) > 500:
                        nl = self.buffer.find("\n", 100)
                        if nl != -1:
                            events.append(
                                StreamEvent(event_type="text", text=self.buffer[: nl + 1])
                            )
                            self.buffer = self.buffer[nl + 1 :]
                            continue
                    break

                # Emit safe prefix of text
                safe_len = self._safe_text_emit_len()
                if safe_len > 0:
                    events.append(StreamEvent(event_type="text", text=self.buffer[:safe_len]))
                    self.buffer = self.buffer[safe_len:]
                break

            elif self.state == "thinking":
                close_idx = self.buffer.find("</think>")
                if close_idx != -1:
                    events.append(
                        StreamEvent(event_type="thinking", thinking=self.buffer[:close_idx])
                    )
                    self.buffer = self.buffer[close_idx + 8 :]
                    self.state = "normal"
                    continue

                if len(self.buffer) > 8:
                    emit_len = len(self.buffer) - 8
                    events.append(
                        StreamEvent(event_type="thinking", thinking=self.buffer[:emit_len])
                    )
                    self.buffer = self.buffer[emit_len:]
                break

            elif self.state == "in_tool_call":
                is_completed = False
                body_end = len(self.buffer)
                after_idx = len(self.buffer)

                if self._current_format == "xml_dsml":
                    close_m = _TAG_CLOSE_RE.search(self.buffer)
                    if close_m:
                        is_completed = True
                        body_end = close_m.start()
                        after_idx = close_m.end()

                elif self._current_format == "native_token":
                    close_m = _NATIVE_TOKEN_END_RE.search(self.buffer)
                    if close_m:
                        is_completed = True
                        body_end = close_m.start()
                        after_idx = close_m.end()

                elif self._current_format == "qwen":
                    close_m = _QWEN_END_RE.search(self.buffer)
                    if close_m:
                        is_completed = True
                        body_end = close_m.start()
                        after_idx = close_m.end()

                elif self._current_format == "markdown_json":
                    if len(self.buffer) >= 3:
                        close_pos = self.buffer[3:].find("```")
                        if close_pos != -1:
                            is_completed = True
                            body_end = 3 + close_pos + 3
                            after_idx = body_end

                elif self._current_format == "react":
                    close_pos = self.buffer.find("\n\n")
                    if close_pos != -1:
                        is_completed = True
                        body_end = close_pos
                        after_idx = close_pos + 2
                    else:
                        thought_pos = self.buffer.find("Thought:")
                        if thought_pos != -1:
                            is_completed = True
                            body_end = thought_pos
                            after_idx = thought_pos

                if is_completed:
                    call_str = self.buffer[:body_end]
                    name = self._current_name or "unknown"
                    if self._current_format == "xml_dsml":
                        synthetic = f'<invoke name="{name}">{call_str}</invoke>'
                    elif self._current_format == "native_token":
                        synthetic = f"<｜tool call begin｜>function={name}<｜tool sep｜>{call_str}<｜tool call end｜>"
                    elif self._current_format == "qwen":
                        synthetic = f"<|action_start|><|action_name|>{name}<|action_args|>{call_str}<|action_end|>"
                    elif self._current_format == "markdown_json":
                        synthetic = call_str
                    elif self._current_format == "react":
                        synthetic = f"Action: {name}\nAction Input: {call_str}"
                    else:
                        synthetic = call_str

                    try:
                        tc = self.parser.parse(synthetic)
                        events.append(StreamEvent(event_type="tool_call_completed", tool_call=tc))
                    except Exception:
                        pass

                    self.buffer = self.buffer[after_idx:]
                    self.state = "normal"
                    self._current_name = None
                    self._current_call_id = None
                    self._current_format = None
                    self._last_emitted_body_len = 0
                    continue

                # Emit incremental argument delta
                reserve = {
                    "xml_dsml": 16,
                    "native_token": 20,
                    "qwen": 16,
                    "markdown_json": 4,
                    "react": 10,
                }.get(self._current_format or "", 10)

                if len(self.buffer) > reserve + self._last_emitted_body_len:
                    emit_end = len(self.buffer) - reserve
                    delta = self.buffer[self._last_emitted_body_len : emit_end]
                    if delta:
                        events.append(
                            StreamEvent(
                                event_type="tool_call_arguments_chunk",
                                name=self._current_name,
                                delta=delta,
                            )
                        )
                    self._last_emitted_body_len = emit_end
                break

        return events

    def finish(self) -> list[StreamEvent]:
        """Flush any pending data and recover truncated invocations on stream completion."""
        events: list[StreamEvent] = []

        if self.state == "thinking":
            if self.buffer:
                events.append(StreamEvent(event_type="thinking", thinking=self.buffer))
        elif self.state == "in_tool_call":
            if self.buffer:
                name = self._current_name or "unknown"
                if self._current_format == "xml_dsml":
                    synthetic = f'<invoke name="{name}">{self.buffer}</invoke>'
                elif self._current_format == "native_token":
                    synthetic = f"<｜tool call begin｜>function={name}<｜tool sep｜>{self.buffer}<｜tool call end｜>"
                elif self._current_format == "qwen":
                    synthetic = f"<|action_start|><|action_name|>{name}<|action_args|>{self.buffer}<|action_end|>"
                elif self._current_format == "markdown_json":
                    synthetic = self.buffer
                elif self._current_format == "react":
                    synthetic = f"Action: {name}\nAction Input: {self.buffer}"
                else:
                    synthetic = self.buffer

                try:
                    tc = self.parser.parse(synthetic)
                    events.append(StreamEvent(event_type="tool_call_completed", tool_call=tc))
                except Exception:
                    pass
        elif self.state == "normal":
            if self.buffer:
                try:
                    tc = self.parser.parse(self.buffer)
                    events.append(StreamEvent(event_type="tool_call_completed", tool_call=tc))
                except Exception:
                    events.append(StreamEvent(event_type="text", text=self.buffer))

        self.reset()
        return events

    def _safe_text_emit_len(self) -> int:
        text = self.buffer
        if not text:
            return 0

        prefixes = (
            "<",
            "<｜",
            "<|",
            "<t",
            "<th",
            "<thi",
            "<thin",
            "<think",
            "<d",
            "<ds",
            "<dsm",
            "<dsml",
            "<i",
            "<in",
            "<inv",
            "<invo",
            "<invok",
            "<invoke",
            "```",
            "``",
            "`",
            "A",
            "Ac",
            "Act",
            "Acti",
            "Actio",
            "Action",
        )

        for p in prefixes:
            if text.endswith(p):
                return max(0, len(text) - len(p))

        return len(text)
