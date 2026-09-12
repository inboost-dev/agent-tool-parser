// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Represents a structured tool invocation extracted from LLM output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub args: serde_json::Value,
    pub raw_source: String,
}

impl ToolCall {
    pub fn new(
        name: impl Into<String>,
        args: serde_json::Value,
        raw_source: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            args,
            raw_source: raw_source.into(),
        }
    }
}

/// Raised when no valid tool call could be extracted or validation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolError {
    pub message: String,
}

impl ToolError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ToolError {}

/// Configuration and state for the tool call parser.
#[derive(Debug, Clone)]
pub struct ToolParserConfig {
    pub allowed_tools: Option<HashSet<String>>,
    pub tool_aliases: HashMap<String, String>,
    pub param_aliases: HashMap<String, String>,
    pub code_param_keys: HashSet<String>,
    pub strip_thinking: bool,
    pub allow_shell_fallback: bool,
}

impl Default for ToolParserConfig {
    fn default() -> Self {
        let mut tool_aliases = HashMap::new();
        let default_tools = [
            ("readfile", "read_file"),
            ("writefile", "write_file"),
            ("strreplace", "str_replace"),
            ("tool_search", "search"),
            ("tool_bash", "bash"),
            ("tool_read_file", "read_file"),
            ("tool_write_file", "write_file"),
            ("tool_str_replace", "str_replace"),
            ("grep", "search"),
            ("grep_search", "search"),
            ("view", "read_file"),
            ("view_file", "read_file"),
            ("edit", "str_replace"),
            ("edit_file", "str_replace"),
            ("sh", "bash"),
            ("shell", "bash"),
            ("terminal", "bash"),
        ];
        for (k, v) in default_tools {
            tool_aliases.insert(k.to_string(), v.to_string());
        }

        let mut param_aliases = HashMap::new();
        let default_params = [
            ("cmd", "command"),
            ("file", "path"),
            ("filepath", "path"),
            ("file_path", "path"),
            ("target_file", "path"),
            ("filename", "path"),
            ("query", "pattern"),
            ("old", "old_str"),
            ("new", "new_str"),
        ];
        for (k, v) in default_params {
            param_aliases.insert(k.to_string(), v.to_string());
        }

        let code_param_keys = [
            "old_str",
            "new_str",
            "content",
            "old",
            "new",
            "code",
            "text",
            "replacement",
            "patch",
            "diff",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            allowed_tools: None,
            tool_aliases,
            param_aliases,
            code_param_keys,
            strip_thinking: true,
            allow_shell_fallback: false,
        }
    }
}
