// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

use crate::models::{ToolCall, ToolParserConfig};
use crate::parser::ToolParser;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

static ATTR_NAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)\b(?:name|tool|tool_name|function)\s*=\s*['"]?([a-zA-Z0-9_.:-]+)['"]?"#)
        .unwrap()
});

static TAG_CLOSE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)</[｜|]*(?:dsml[｜|]*)?(?:tool_invoke|invoke|tool_call|call|tool|invocation|function_call|function|action|tool_use|ant_tool_use|function_use|действие|вызов_функции|функция|инструмент)(?::\s*[\w-]+)?\s*>"#).unwrap()
});

static NATIVE_TOKEN_START_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?s)<[｜|]*tool call begin[｜|]*>(?:function=)?(?P<name>[\w.:-]+)(?:<[｜|]*tool sep[｜|]*>)?"#).unwrap()
});

static NATIVE_TOKEN_END_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?s)<[｜|]*tool call end[｜|]*>"#).unwrap());

static QWEN_START_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<\|action_start\|><\|action_name\|>(?P<name>[\w.:-]+)<\|action_args\|>"#)
        .unwrap()
});

static QWEN_END_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"(?is)<\|action_end\|>"#).unwrap());

static REACT_START_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)Action:\s*(?P<name>[\w.:-]+)\s*\n(?:Action Input:\s*)?"#).unwrap()
});

static CODEBLOCK_START_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?s)```(?:json)?\s*\{"#).unwrap());

static JSON_NAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)["'](?:name|tool|action|function)["']\s*:\s*["'](?P<name>[\w.:-]+)["']"#)
        .unwrap()
});

/// Events emitted by `StreamingToolParser` during token-by-token stream consumption.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum StreamEvent {
    /// Plain text outside tool calls and reasoning blocks.
    Text(String),
    /// Deliberation text emitted inside `<think>...</think>`.
    Thinking(String),
    /// Immediate notification that a tool invocation has started, emitted as soon as
    /// the tool name is resolved (enabling early sandbox warmup and permission checks).
    ToolCallStarted {
        name: String,
        call_id: Option<String>,
    },
    /// Incremental argument payload chunk for the currently active tool call.
    ToolCallArgumentsChunk { name: String, delta: String },
    /// The tool invocation has completed and arguments are fully parsed and validated.
    ToolCallCompleted(ToolCall),
}

#[derive(Debug, Clone, PartialEq)]
enum StreamState {
    Normal,
    InThinking,
    InToolCall {
        name: String,
        call_id: Option<String>,
        format: ToolStreamFormat,
        body_start_in_buf: usize,
        last_emitted_body_len: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolStreamFormat {
    XmlDsml,
    NativeToken,
    Qwen,
    MarkdownJson,
    ReAct,
}

/// Token-by-token streaming tool parser.
/// Processes incremental text chunks from SSE / LLM streaming connections, emitting
/// real-time events (`ToolCallStarted`, `ToolCallArgumentsChunk`, `ToolCallCompleted`, `Thinking`, `Text`).
pub struct StreamingToolParser {
    config: ToolParserConfig,
    parser: ToolParser,
    buffer: String,
    state: StreamState,
}

impl Default for StreamingToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingToolParser {
    pub fn new() -> Self {
        Self::with_config(ToolParserConfig::default())
    }

    pub fn with_config(config: ToolParserConfig) -> Self {
        let parser = ToolParser::with_config(config.clone());
        Self {
            config,
            parser,
            buffer: String::new(),
            state: StreamState::Normal,
        }
    }

    /// Access the underlying configuration.
    pub fn config(&self) -> &ToolParserConfig {
        &self.config
    }

    /// Reset internal parser state and buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.state = StreamState::Normal;
    }

    /// Feed an incremental text chunk into the streaming parser and retrieve any resulting events.
    pub fn feed(&mut self, chunk: &str) -> Vec<StreamEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        loop {
            match &self.state {
                StreamState::Normal => {
                    // 1. Check for thinking block <think>
                    if let Some(idx) = self.buffer.find("<think>") {
                        if idx > 0 {
                            events.push(StreamEvent::Text(self.buffer[..idx].to_string()));
                        }
                        self.buffer = self.buffer[idx + 7..].to_string();
                        self.state = StreamState::InThinking;
                        continue;
                    }

                    // 2. Check for XML / DSML invoke tag
                    static INVOKE_OPEN_RE: Lazy<Regex> = Lazy::new(|| {
                        Regex::new(r#"(?is)<[｜|]*(?:dsml[｜|]*)?(?P<tag>tool_invoke|invoke|tool_call|call|tool|invocation|function_call|function|action|tool_use|ant_tool_use|function_use|действие|вызов_функции|функция|инструмент)(?::\s*(?P<colon_tool>[\w-]+))?\b(?P<attrs>[^>]*)>"#).unwrap()
                    });

                    if let Some(caps) = INVOKE_OPEN_RE.captures(&self.buffer) {
                        let m = caps.get(0).unwrap();
                        let start = m.start();
                        let end = m.end();
                        let colon_tool = caps.name("colon_tool").map(|c| c.as_str());
                        let attrs = caps.name("attrs").map(|c| c.as_str()).unwrap_or("");

                        let raw_name = if let Some(ct) = colon_tool {
                            if !ct.is_empty() {
                                Some(ct.to_string())
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        let name_opt = raw_name.or_else(|| {
                            ATTR_NAME_RE
                                .captures(attrs)
                                .map(|c| c.get(1).unwrap().as_str().trim().to_string())
                        });

                        if let Some(name) = name_opt {
                            let norm_name = self.parser.normalize_name(&name);
                            if start > 0 {
                                events.push(StreamEvent::Text(self.buffer[..start].to_string()));
                            }

                            events.push(StreamEvent::ToolCallStarted {
                                name: norm_name.clone(),
                                call_id: None,
                            });

                            self.buffer = self.buffer[end..].to_string();
                            self.state = StreamState::InToolCall {
                                name: norm_name,
                                call_id: None,
                                format: ToolStreamFormat::XmlDsml,
                                body_start_in_buf: 0,
                                last_emitted_body_len: 0,
                            };
                            continue;
                        }
                    }

                    // 3. Check for Native DeepSeek token
                    if let Some(caps) = NATIVE_TOKEN_START_RE.captures(&self.buffer) {
                        let m = caps.get(0).unwrap();
                        let start = m.start();
                        let end = m.end();
                        let name = caps.name("name").unwrap().as_str().trim();
                        let norm_name = self.parser.normalize_name(name);

                        if start > 0 {
                            events.push(StreamEvent::Text(self.buffer[..start].to_string()));
                        }

                        events.push(StreamEvent::ToolCallStarted {
                            name: norm_name.clone(),
                            call_id: None,
                        });

                        self.buffer = self.buffer[end..].to_string();
                        self.state = StreamState::InToolCall {
                            name: norm_name,
                            call_id: None,
                            format: ToolStreamFormat::NativeToken,
                            body_start_in_buf: 0,
                            last_emitted_body_len: 0,
                        };
                        continue;
                    }

                    // 4. Check for Qwen Action token
                    if let Some(caps) = QWEN_START_RE.captures(&self.buffer) {
                        let m = caps.get(0).unwrap();
                        let start = m.start();
                        let end = m.end();
                        let name = caps.name("name").unwrap().as_str().trim();
                        let norm_name = self.parser.normalize_name(name);

                        if start > 0 {
                            events.push(StreamEvent::Text(self.buffer[..start].to_string()));
                        }

                        events.push(StreamEvent::ToolCallStarted {
                            name: norm_name.clone(),
                            call_id: None,
                        });

                        self.buffer = self.buffer[end..].to_string();
                        self.state = StreamState::InToolCall {
                            name: norm_name,
                            call_id: None,
                            format: ToolStreamFormat::Qwen,
                            body_start_in_buf: 0,
                            last_emitted_body_len: 0,
                        };
                        continue;
                    }

                    // 5. Check for ReAct notation
                    if let Some(caps) = REACT_START_RE.captures(&self.buffer) {
                        let m = caps.get(0).unwrap();
                        let start = m.start();
                        let end = m.end();
                        let name = caps.name("name").unwrap().as_str().trim();
                        let norm_name = self.parser.normalize_name(name);

                        if start > 0 {
                            events.push(StreamEvent::Text(self.buffer[..start].to_string()));
                        }

                        events.push(StreamEvent::ToolCallStarted {
                            name: norm_name.clone(),
                            call_id: None,
                        });

                        self.buffer = self.buffer[end..].to_string();
                        self.state = StreamState::InToolCall {
                            name: norm_name,
                            call_id: None,
                            format: ToolStreamFormat::ReAct,
                            body_start_in_buf: 0,
                            last_emitted_body_len: 0,
                        };
                        continue;
                    }

                    // 6. Check for Markdown JSON block
                    if let Some(cb_idx) = self.buffer.find("```") {
                        if let Some(caps) = CODEBLOCK_START_RE.captures(&self.buffer[cb_idx..]) {
                            let m = caps.get(0).unwrap();
                            let after_start = &self.buffer[cb_idx + m.end()..];
                            if let Some(ncaps) = JSON_NAME_RE.captures(after_start) {
                                let name = ncaps.name("name").unwrap().as_str().trim();
                                let norm_name = self.parser.normalize_name(name);

                                if cb_idx > 0 {
                                    events
                                        .push(StreamEvent::Text(self.buffer[..cb_idx].to_string()));
                                }

                                events.push(StreamEvent::ToolCallStarted {
                                    name: norm_name.clone(),
                                    call_id: None,
                                });

                                self.buffer = self.buffer[cb_idx..].to_string();
                                self.state = StreamState::InToolCall {
                                    name: norm_name,
                                    call_id: None,
                                    format: ToolStreamFormat::MarkdownJson,
                                    body_start_in_buf: 0,
                                    last_emitted_body_len: 0,
                                };
                                continue;
                            }
                        }

                        // If not matched as tool call yet, check if it closed
                        let after_fence = &self.buffer[cb_idx + 3..];
                        if let Some(close_pos) = after_fence.find("```") {
                            let full_code_end = cb_idx + 3 + close_pos + 3;
                            events
                                .push(StreamEvent::Text(self.buffer[..full_code_end].to_string()));
                            self.buffer = self.buffer[full_code_end..].to_string();
                            continue;
                        }

                        // Still open, might become a tool call or close soon.
                        if cb_idx > 0 {
                            events.push(StreamEvent::Text(self.buffer[..cb_idx].to_string()));
                            self.buffer = self.buffer[cb_idx..].to_string();
                        }

                        if self.buffer.len() > 500 {
                            if let Some(nl) = self.buffer[100..].find('\n') {
                                let cut = 100 + nl + 1;
                                events.push(StreamEvent::Text(self.buffer[..cut].to_string()));
                                self.buffer = self.buffer[cut..].to_string();
                                continue;
                            }
                        }

                        break;
                    }

                    // If buffer does not contain partial delimiter, emit safe text prefix
                    let safe_len = self.safe_text_emit_len();
                    if safe_len > 0 {
                        events.push(StreamEvent::Text(self.buffer[..safe_len].to_string()));
                        self.buffer = self.buffer[safe_len..].to_string();
                    }
                    break;
                }

                StreamState::InThinking => {
                    if let Some(idx) = self.buffer.find("</think>") {
                        events.push(StreamEvent::Thinking(self.buffer[..idx].to_string()));
                        self.buffer = self.buffer[idx + 8..].to_string();
                        self.state = StreamState::Normal;
                        continue;
                    }

                    // Emit thinking delta up to potential tag prefix "</think>" (max 8 chars)
                    if self.buffer.len() > 8 {
                        let emit_len = self.buffer.len() - 8;
                        events.push(StreamEvent::Thinking(self.buffer[..emit_len].to_string()));
                        self.buffer = self.buffer[emit_len..].to_string();
                    }
                    break;
                }

                StreamState::InToolCall {
                    name,
                    call_id,
                    format,
                    body_start_in_buf: _,
                    last_emitted_body_len,
                } => {
                    let (is_completed, body_end, after_call_idx) = match format {
                        ToolStreamFormat::XmlDsml => {
                            if let Some(m) = TAG_CLOSE_RE.find(&self.buffer) {
                                (true, m.start(), m.end())
                            } else {
                                (false, self.buffer.len(), self.buffer.len())
                            }
                        }
                        ToolStreamFormat::NativeToken => {
                            if let Some(m) = NATIVE_TOKEN_END_RE.find(&self.buffer) {
                                (true, m.start(), m.end())
                            } else {
                                (false, self.buffer.len(), self.buffer.len())
                            }
                        }
                        ToolStreamFormat::Qwen => {
                            if let Some(m) = QWEN_END_RE.find(&self.buffer) {
                                (true, m.start(), m.end())
                            } else {
                                (false, self.buffer.len(), self.buffer.len())
                            }
                        }
                        ToolStreamFormat::MarkdownJson => {
                            if self.buffer.len() >= 3 {
                                if let Some(pos) = self.buffer[3..].find("```") {
                                    let abs_end = 3 + pos + 3;
                                    (true, abs_end, abs_end)
                                } else {
                                    (false, self.buffer.len(), self.buffer.len())
                                }
                            } else {
                                (false, self.buffer.len(), self.buffer.len())
                            }
                        }
                        ToolStreamFormat::ReAct => {
                            if let Some(pos) = self.buffer.find("\n\n") {
                                (true, pos, pos + 2)
                            } else if let Some(pos) = self.buffer.find("Thought:") {
                                (true, pos, pos)
                            } else {
                                (false, self.buffer.len(), self.buffer.len())
                            }
                        }
                    };

                    if is_completed {
                        let call_str = &self.buffer[..body_end];
                        // Construct synthetic full string for parsing if needed
                        let synthetic = match format {
                            ToolStreamFormat::XmlDsml => {
                                format!(r#"<invoke name="{}">{}</invoke>"#, name, call_str)
                            }
                            ToolStreamFormat::NativeToken => {
                                format!(
                                    r#"<｜tool call begin｜>function={}<｜tool sep｜>{}<｜tool call end｜>"#,
                                    name, call_str
                                )
                            }
                            ToolStreamFormat::Qwen => {
                                format!(
                                    r#"<|action_start|><|action_name|>{}<|action_args|>{}<|action_end|>"#,
                                    name, call_str
                                )
                            }
                            ToolStreamFormat::MarkdownJson => call_str.to_string(),
                            ToolStreamFormat::ReAct => {
                                format!("Action: {}\nAction Input: {}", name, call_str)
                            }
                        };

                        if let Ok(tc) = self.parser.parse(&synthetic) {
                            events.push(StreamEvent::ToolCallCompleted(tc));
                        }

                        self.buffer = self.buffer[after_call_idx..].to_string();
                        self.state = StreamState::Normal;
                        continue;
                    }

                    // If not completed, emit incremental argument delta
                    let reserve = match format {
                        ToolStreamFormat::XmlDsml => 16,
                        ToolStreamFormat::NativeToken => 20,
                        ToolStreamFormat::Qwen => 16,
                        ToolStreamFormat::MarkdownJson => 4,
                        ToolStreamFormat::ReAct => 10,
                    };

                    let char_count = self.buffer.chars().count();
                    if char_count > reserve + *last_emitted_body_len {
                        let emit_char_end = char_count - reserve;
                        let last_byte_start = self
                            .buffer
                            .char_indices()
                            .nth(*last_emitted_body_len)
                            .map(|(idx, _)| idx)
                            .unwrap_or(self.buffer.len());
                        let emit_byte_end = self
                            .buffer
                            .char_indices()
                            .nth(emit_char_end)
                            .map(|(idx, _)| idx)
                            .unwrap_or(self.buffer.len());

                        let delta = &self.buffer[last_byte_start..emit_byte_end];
                        if !delta.is_empty() {
                            events.push(StreamEvent::ToolCallArgumentsChunk {
                                name: name.clone(),
                                delta: delta.to_string(),
                            });
                        }
                        self.state = StreamState::InToolCall {
                            name: name.clone(),
                            call_id: call_id.clone(),
                            format: *format,
                            body_start_in_buf: 0,
                            last_emitted_body_len: emit_char_end,
                        };
                    }
                    break;
                }
            }
        }

        events
    }

    /// Flush and finish stream consumption, recovering any unclosed/truncated invocations.
    pub fn finish(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        match &self.state {
            StreamState::InThinking => {
                if !self.buffer.is_empty() {
                    events.push(StreamEvent::Thinking(self.buffer.clone()));
                }
            }
            StreamState::InToolCall { name, format, .. } => {
                if !self.buffer.is_empty() {
                    let synthetic = match format {
                        ToolStreamFormat::XmlDsml => {
                            format!(r#"<invoke name="{}">{}</invoke>"#, name, self.buffer)
                        }
                        ToolStreamFormat::NativeToken => {
                            format!(
                                r#"<｜tool call begin｜>function={}<｜tool sep｜>{}<｜tool call end｜>"#,
                                name, self.buffer
                            )
                        }
                        ToolStreamFormat::Qwen => {
                            format!(
                                r#"<|action_start|><|action_name|>{}<|action_args|>{}<|action_end|>"#,
                                name, self.buffer
                            )
                        }
                        ToolStreamFormat::MarkdownJson => {
                            let mut synthetic = self.buffer.clone();
                            if !synthetic.trim_end().ends_with("```") {
                                synthetic.push_str("\n```");
                            }
                            synthetic
                        }
                        ToolStreamFormat::ReAct => {
                            format!("Action: {}\nAction Input: {}", name, self.buffer)
                        }
                    };

                    if let Ok(tc) = self.parser.parse(&synthetic) {
                        events.push(StreamEvent::ToolCallCompleted(tc));
                    }
                }
            }
            StreamState::Normal => {
                if !self.buffer.is_empty() {
                    if let Ok(tc) = self.parser.parse(&self.buffer) {
                        events.push(StreamEvent::ToolCallCompleted(tc));
                    } else {
                        events.push(StreamEvent::Text(self.buffer.clone()));
                    }
                }
            }
        }

        self.reset();
        events
    }

    fn safe_text_emit_len(&self) -> usize {
        let text = &self.buffer;
        if text.is_empty() {
            return 0;
        }

        // Check if ends with partial opening delimiters
        let prefixes = [
            "<", "<｜", "<|", "<t", "<th", "<thi", "<thin", "<think", "<d", "<ds", "<dsm", "<dsml",
            "<i", "<in", "<inv", "<invo", "<invok", "<invoke", "```", "``", "`", "A", "Ac", "Act",
            "Acti", "Actio", "Action",
        ];

        for p in &prefixes {
            if text.ends_with(p) {
                return text.len().saturating_sub(p.len());
            }
        }

        text.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_tool_call_dsml() {
        let mut parser = StreamingToolParser::new();

        let chunk1 = "Thinking about files...\n<think>analyzing</think>\nI will now read:\n<｜DSML｜invoke name=\"read_file\">";
        let events1 = parser.feed(chunk1);

        assert!(events1
            .iter()
            .any(|e| matches!(e, StreamEvent::Text(t) if t.contains("Thinking about files"))));
        assert!(events1
            .iter()
            .any(|e| matches!(e, StreamEvent::Thinking(t) if t == "analyzing")));
        assert!(events1.iter().any(
            |e| matches!(e, StreamEvent::ToolCallStarted { name, .. } if name == "read_file")
        ));

        let chunk2 = "<｜DSML｜parameter name=\"path\">src/lib.rs</｜DSML｜parameter>\n</｜DSML｜invoke>\nFinished!";
        let events2 = parser.feed(chunk2);

        let completed = events2.iter().find_map(|e| match e {
            StreamEvent::ToolCallCompleted(tc) => Some(tc),
            _ => None,
        });
        assert!(completed.is_some());
        let tc = completed.unwrap();
        assert_eq!(tc.name, "read_file");
        assert_eq!(tc.args["path"], "src/lib.rs");

        assert!(events2
            .iter()
            .any(|e| matches!(e, StreamEvent::Text(t) if t.contains("Finished!"))));
    }

    #[test]
    fn test_streaming_json_markdown() {
        let mut parser = StreamingToolParser::new();
        let chunk1 = "Action required:\n```json\n{\"tool\": \"bash\", ";
        let events1 = parser.feed(chunk1);
        assert!(events1
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallStarted { name, .. } if name == "bash")));

        let chunk2 = "\"command\": \"cargo test\"}\n```\nAll done.";
        let events2 = parser.feed(chunk2);
        let completed = events2.iter().find_map(|e| match e {
            StreamEvent::ToolCallCompleted(tc) => Some(tc),
            _ => None,
        });
        assert!(completed.is_some());
        let tc = completed.unwrap();
        assert_eq!(tc.name, "bash");
        assert_eq!(tc.args["command"], "cargo test");
    }

    #[test]
    fn test_streaming_finish_truncated() {
        let mut parser = StreamingToolParser::new();
        let chunk = "<｜DSML｜invoke name=\"search\"><｜DSML｜parameter name=\"q\">rust";
        let events = parser.feed(chunk);
        assert!(events
            .iter()
            .any(|e| matches!(e, StreamEvent::ToolCallStarted { name, .. } if name == "search")));

        // Stream cut off by EOF
        let finished = parser.finish();
        let completed = finished.iter().find_map(|e| match e {
            StreamEvent::ToolCallCompleted(tc) => Some(tc),
            _ => None,
        });
        assert!(completed.is_some());
        assert_eq!(completed.unwrap().name, "search");
    }
}
