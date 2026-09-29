// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

use crate::cleaners::{clean_param_val, extract_json_objects, safe_json_loads, strip_thinking};
use crate::models::{ToolCall, ToolError};
use crate::parser::{extract_xml_parameters, INVOKE_RE};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

static ATTR_NAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)\b(?:name|tool|tool_name|function)\s*=\s*['"]?([a-zA-Z0-9_.:-]+)['"]?"#)
        .unwrap()
});

static NATIVE_DEEPSEEK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?s)<[｜|]*tool call begin[｜|]*>(?:function=)?(?P<name>[\w.:-]+)<[｜|]*tool sep[｜|]*>(?P<args>.*?)<[｜|]*tool call end[｜|]*>"#).unwrap()
});

static QWEN_ACTION_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<\|action_start\|><\|action_name\|>(?P<name>[\w.:-]+)<\|action_args\|>(?P<args>.*?)<\|action_end\|>"#).unwrap()
});

static REACT_ACTION_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)Action:\s*(?P<name>[\w.:-]+)\s*\nAction Input:\s*(?P<args>[^\n]+(?:\n[^\n]+)*)"#,
    )
    .unwrap()
});

static MISTRAL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?is)\[TOOL_CALLS\]\s*(\[\s*\{.*?\}\s*\])"#).unwrap());

/// Zero-copy capable raw representation of an extracted tool call.
/// Holds the unparsed arguments as a raw string slice/String to allow high-throughput
/// structural inspection, routing, and filtering before incurring full JSON deserialization costs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawToolCall {
    pub name: String,
    pub raw_args: String,
    pub raw_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
}

impl RawToolCall {
    pub fn new(
        name: impl Into<String>,
        raw_args: impl Into<String>,
        raw_source: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            raw_args: raw_args.into(),
            raw_source: raw_source.into(),
            call_id: None,
        }
    }

    pub fn with_call_id(mut self, call_id: impl Into<String>) -> Self {
        self.call_id = Some(call_id.into());
        self
    }

    /// Parse raw arguments into structured key-value map or Value on demand.
    pub fn parse_args(&self) -> Result<Value, ToolError> {
        let trimmed = self.raw_args.trim();

        // 1. Check if raw_args contains XML/DSML parameter tags
        let xml_params = extract_xml_parameters(trimmed);
        if !xml_params.is_empty() {
            let mut args_map = Map::new();
            for (pname, val) in xml_params {
                let is_code =
                    pname.contains("code") || pname.contains("content") || pname.contains("script");
                let clean_val = clean_param_val(&val, is_code);
                let val_json = if let Some(j) = safe_json_loads(&clean_val) {
                    j
                } else {
                    Value::String(clean_val)
                };
                args_map.insert(pname, val_json);
            }
            return Ok(Value::Object(args_map));
        }

        // 2. Try JSON deserialization
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            if let Some(val) = safe_json_loads(trimmed) {
                return Ok(val);
            }
            return Err(ToolError::new(format!(
                "Failed to parse JSON arguments for tool '{}'",
                self.name
            )));
        }

        // 3. Try Python kwargs
        if trimmed.contains('=') || trimmed.contains('"') || trimmed.contains('\'') {
            let py_kwargs = crate::python_calls::parse_python_kwargs(trimmed);
            if !py_kwargs.is_empty() {
                return Ok(Value::Object(py_kwargs));
            }
        }

        // 4. Fallback for string / command payloads
        let mut map = Map::new();

        let key =
            if self.name.contains("bash") || self.name.contains("sh") || self.name.contains("exec")
            {
                "command"
            } else {
                "input"
            };
        map.insert(key.to_string(), Value::String(trimmed.to_string()));
        Ok(Value::Object(map))
    }

    /// Convert into a full ToolCall instance.
    pub fn to_tool_call(&self) -> Result<ToolCall, ToolError> {
        let args = self.parse_args()?;
        Ok(ToolCall::new(
            self.name.clone(),
            args,
            self.raw_source.clone(),
        ))
    }
}

/// Extract raw tool calls from text without full JSON deserialization overhead.
pub fn extract_raw_tool_calls(text: &str) -> Vec<RawToolCall> {
    let clean = strip_thinking(text);
    let mut results = Vec::new();

    // 1. DSML and XML invocation blocks
    for caps in INVOKE_RE.captures_iter(&clean) {
        let full = caps.get(0).unwrap().as_str();
        let colon_tool = caps.name("colon_tool").map(|m| m.as_str().trim());
        let attrs = caps.name("attrs").map(|m| m.as_str()).unwrap_or("");
        let body = caps.name("body").map(|m| m.as_str()).unwrap_or("");

        let name_opt = if let Some(ct) = colon_tool {
            if !ct.is_empty() {
                Some(ct.to_string())
            } else {
                None
            }
        } else {
            None
        };

        let name = name_opt.or_else(|| {
            ATTR_NAME_RE
                .captures(attrs)
                .map(|c| c.get(1).unwrap().as_str().trim().to_string())
        });

        if let Some(tname) = name {
            if !tname.is_empty() {
                results.push(RawToolCall::new(tname, body.trim(), full));
            }
        }
    }

    if !results.is_empty() {
        return results;
    }

    // 2. Native DeepSeek tokens
    for caps in NATIVE_DEEPSEEK_RE.captures_iter(&clean) {
        let full = caps.get(0).unwrap().as_str();
        let name = caps.name("name").unwrap().as_str().trim();
        let args = caps.name("args").unwrap().as_str().trim();
        if !name.is_empty() {
            results.push(RawToolCall::new(name, args, full));
        }
    }

    if !results.is_empty() {
        return results;
    }

    // 3. Qwen action tokens
    for caps in QWEN_ACTION_RE.captures_iter(&clean) {
        let full = caps.get(0).unwrap().as_str();
        let name = caps.name("name").unwrap().as_str().trim();
        let args = caps.name("args").unwrap().as_str().trim();
        if !name.is_empty() {
            results.push(RawToolCall::new(name, args, full));
        }
    }

    if !results.is_empty() {
        return results;
    }

    // 4. Mistral [TOOL_CALLS]
    if let Some(caps) = MISTRAL_RE.captures(&clean) {
        let full = caps.get(0).unwrap().as_str();
        let arr_str = caps.get(1).unwrap().as_str();
        if let Some(Value::Array(items)) = safe_json_loads(arr_str) {
            for item in items {
                if let Value::Object(obj) = item {
                    if let Some(name_val) = obj.get("name").and_then(|v| v.as_str()) {
                        let raw_args = if let Some(args_val) = obj.get("arguments") {
                            if let Some(s) = args_val.as_str() {
                                s.to_string()
                            } else {
                                serde_json::to_string(args_val).unwrap_or_default()
                            }
                        } else {
                            String::new()
                        };
                        results.push(RawToolCall::new(name_val, raw_args, full));
                    }
                }
            }
        }
    }

    if !results.is_empty() {
        return results;
    }

    // 5. ReAct pattern
    if let Some(caps) = REACT_ACTION_RE.captures(&clean) {
        let full = caps.get(0).unwrap().as_str();
        let name = caps.name("name").unwrap().as_str().trim();
        let args = caps.name("args").unwrap().as_str().trim();
        if !name.is_empty() {
            results.push(RawToolCall::new(name, args, full));
        }
    }

    if !results.is_empty() {
        return results;
    }

    // 6. JSON objects in Markdown or raw text
    let json_objs = extract_json_objects(&clean);
    for raw_json in json_objs {
        if let Some(Value::Object(map)) = safe_json_loads(&raw_json) {
            // Check standard OpenAI tool_calls structure
            if let Some(Value::Array(calls)) = map.get("tool_calls") {
                for c in calls {
                    if let Value::Object(call_obj) = c {
                        let call_id = call_obj
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        if let Some(Value::Object(func_obj)) = call_obj.get("function") {
                            if let Some(name) = func_obj.get("name").and_then(|v| v.as_str()) {
                                let raw_args = func_obj
                                    .get("arguments")
                                    .map(|v| {
                                        if let Some(s) = v.as_str() {
                                            s.to_string()
                                        } else {
                                            serde_json::to_string(v).unwrap_or_default()
                                        }
                                    })
                                    .unwrap_or_default();
                                let mut raw_call = RawToolCall::new(name, raw_args, &raw_json);
                                raw_call.call_id = call_id;
                                results.push(raw_call);
                            }
                        }
                    }
                }
                if !results.is_empty() {
                    return results;
                }
            }

            // Direct object with tool name key
            let name_candidate = map
                .get("name")
                .or_else(|| map.get("tool"))
                .or_else(|| map.get("action"))
                .or_else(|| map.get("function"))
                .and_then(|v| v.as_str());

            if let Some(name) = name_candidate {
                let raw_args = map
                    .get("arguments")
                    .or_else(|| map.get("parameters"))
                    .or_else(|| map.get("args"))
                    .map(|v| {
                        if let Some(s) = v.as_str() {
                            s.to_string()
                        } else {
                            serde_json::to_string(v).unwrap_or_default()
                        }
                    })
                    .unwrap_or_else(|| raw_json.clone());

                results.push(RawToolCall::new(name, raw_args, raw_json));
            }
        }
    }

    if !results.is_empty() {
        return results;
    }

    // 7. Python function calls in Markdown or raw text
    let py_calls = crate::python_calls::extract_python_function_calls(
        &clean,
        &crate::models::ToolParserConfig::default(),
    );
    for pc in py_calls {
        results.push(RawToolCall::new(pc.name, pc.raw_args, pc.raw_source));
    }

    results
}

/// Try to extract a single raw tool call from text.
pub fn try_extract_raw_tool_call(text: &str) -> Option<RawToolCall> {
    extract_raw_tool_calls(text).into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_raw_tool_call_dsml() {
        let text = r#"<｜DSML｜invoke name="read_file">
<｜DSML｜parameter name="path">src/main.rs</｜DSML｜parameter>
<｜DSML｜parameter name="offset">100</｜DSML｜parameter>
</｜DSML｜invoke>"#;

        let raw_calls = extract_raw_tool_calls(text);
        assert_eq!(raw_calls.len(), 1);
        assert_eq!(raw_calls[0].name, "read_file");
        assert!(raw_calls[0].raw_args.contains("src/main.rs"));

        // Parse on demand
        let parsed = raw_calls[0].parse_args().unwrap();
        assert_eq!(parsed["path"], "src/main.rs");
        assert_eq!(parsed["offset"], 100);

        let call = raw_calls[0].to_tool_call().unwrap();
        assert_eq!(call.name, "read_file");
        assert_eq!(call.args["path"], "src/main.rs");
    }

    #[test]
    fn test_extract_raw_tool_call_json() {
        let text = r#"```json
{"name": "search", "arguments": {"query": "rust", "limit": 10}}
```"#;
        let raw = try_extract_raw_tool_call(text).unwrap();
        assert_eq!(raw.name, "search");
        let args = raw.parse_args().unwrap();
        assert_eq!(args["query"], "rust");
        assert_eq!(args["limit"], 10);
    }

    #[test]
    fn test_extract_raw_multiple_direct_json() {
        let text = r#"I will run two commands:
{"name": "read_file", "arguments": {"path": "main.rs"}}
{"name": "bash", "arguments": {"command": "cargo test"}}"#;
        let calls = extract_raw_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[1].name, "bash");
    }
}
