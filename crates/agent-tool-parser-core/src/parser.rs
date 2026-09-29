// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

use crate::cleaners::{clean_param_val, extract_json_objects, safe_json_loads, strip_thinking};
use crate::models::{ToolCall, ToolError, ToolParserConfig};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

static JSON_BLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)```(?:json)?\s*(\{.*?\})\s*```").unwrap());

static JSON_ARRAY_BLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)```(?:json)?\s*(\[.*?\])\s*```").unwrap());

pub(crate) static INVOKE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)<[｜|]*(?:dsml[｜|]*)?(?P<tag>tool_invoke|invoke|tool_call|call|tool|invocation|function_call|function|action|tool_use|ant_tool_use|function_use|действие|вызов_функции|функция|инструмент)(?::\s*(?P<colon_tool>[\w-]+))?\b(?P<attrs>[^>]*)>(?P<body>.*?)(?:</[｜|]*(?:dsml[｜|]*)?(?:tool_invoke|invoke|tool_call|call|tool|invocation|function_call|function|action|tool_use|ant_tool_use|function_use|действие|вызов_функции|функция|инструмент)(?::\s*[\w-]+)?\s*>|<[｜|]*(?:dsml[｜|]*)?(?:tool_invoke|invoke|tool_call|call|tool|invocation|function_call|function|action|tool_use|ant_tool_use|function_use|действие|вызов_функции|функция|инструмент)\b|</[｜|]*(?:dsml[｜|]*)?(?:tool_calls|function_calls|calls|tools)\s*>|$)"#
    ).unwrap()
});

static PARAM_START_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)<[｜|]*(?:dsml[｜|]*)?(?:parameter|param|arg|argument|параметр|аргумент)\b[^>]*?\bname\s*=\s*['"]?(?P<pname>[\w-]+)['"]?[^>]*>"#
    ).unwrap()
});

static PARAM_END_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)</[｜|]*(?:dsml[｜|]*)?(?:parameter|param|arg|argument|параметр|аргумент)\s*>"#,
    )
    .unwrap()
});

static INVOKE_END_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)</[｜|]*(?:dsml[｜|]*)?(?:tool_invoke|invoke|tool_call|call|tool|invocation|function_call|function|action|tool_use|ant_tool_use|function_use|действие|вызов_функции|функция|инструмент)(?::\s*[a-zA-Z0-9_-]+)?\s*>|</[｜|]*(?:dsml[｜|]*)?(?:tool_calls|function_calls|calls|tools)\s*>"#
    ).unwrap()
});

static PARAM_END_FALLBACK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)</[｜|]+(?:dsml[｜|]*)?[\w:-]+\s*>|</dsml:[\w:-]+\s*>|</(?:tool_name|function_name|tool|invoke|parameter|param|arg|argument|параметр|аргумент|действие)\s*>"#
    ).unwrap()
});

pub(crate) fn extract_xml_parameters(body: &str) -> Vec<(String, String)> {
    let mut cdata_spans: Vec<(usize, usize)> = Vec::new();
    let mut cur = 0;
    while let Some(start) = body[cur..].find("<![CDATA[") {
        let abs_start = cur + start;
        if let Some(end) = body[abs_start..].find("]]>") {
            let abs_end = abs_start + end + 3;
            cdata_spans.push((abs_start, abs_end));
            cur = abs_end;
        } else {
            cdata_spans.push((abs_start, body.len()));
            break;
        }
    }

    let mut results = Vec::new();
    let starts: Vec<(usize, usize, String)> = PARAM_START_RE
        .captures_iter(body)
        .filter_map(|caps| {
            let m = caps.get(0).unwrap();
            let start = m.start();
            if cdata_spans
                .iter()
                .any(|(cs, ce)| start >= *cs && start < *ce)
            {
                return None;
            }
            let pname = caps.name("pname").unwrap().as_str().trim().to_lowercase();
            Some((start, m.end(), pname))
        })
        .collect();

    for i in 0..starts.len() {
        let (_, val_start, ref pname) = starts[i];
        let upper_bound = if i + 1 < starts.len() {
            starts[i + 1].0
        } else {
            body.len()
        };

        let slice = &body[val_start..upper_bound];
        let search_from = if let Some(cdata_start) = slice.find("<![CDATA[") {
            if let Some(cdata_end) = slice[cdata_start..].find("]]>") {
                cdata_start + cdata_end + 3
            } else {
                slice.len()
            }
        } else {
            0
        };

        let val = if search_from < slice.len() {
            if let Some(m) = PARAM_END_RE.find(&slice[search_from..]) {
                &slice[..(search_from + m.start())]
            } else if let Some(m) = INVOKE_END_RE.find(&slice[search_from..]) {
                &slice[..(search_from + m.start())]
            } else if let Some(m) = PARAM_END_FALLBACK_RE.find(&slice[search_from..]) {
                &slice[..(search_from + m.start())]
            } else {
                slice
            }
        } else {
            slice
        };

        results.push((pname.clone(), val.to_string()));
    }

    results
}

static QWEN_AGENT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)(?:<[｜|]*action_start[｜|]*>\s*)?<[｜|]*action_name[｜|]*>\s*(?P<name>[\w-]+)\s*<[｜|]*action_args[｜|]*>\s*(?P<args>.*?)(?:<[｜|]*action_end[｜|]*>|$)"#
    ).unwrap()
});

static CHATGLM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)(?:✿|\b)FUNCTION(?:✿|\b)\s*:?\s*(?P<name>[\w-]+)\s*(?:✿|\b)ARGS(?:✿|\b)\s*:?\s*(?P<args>.*)"#
    ).unwrap()
});

static MISTRAL_TOOL_CALL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?is)\[TOOL_CALLS?\]\s*(?P<body>\[.*?\]|\{.*?\})"#).unwrap());

static LLAMA_PYTHON_TAG_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<[｜|]*python_tag[｜|]*>(?P<body>.*?)(?:<[｜|]*python_tag[｜|]*>|<[｜|]*eom_id[｜|]*>|<[｜|]*eot_id[｜|]*>|$)"#).unwrap()
});

static REACT_HEADER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?im)(?:^|\n)\s*Action:\s*(?P<name>[\w-]+)\s*\n\s*Action\s+Input:\s*"#).unwrap()
});

static DEEPSEEK_NATIVE_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<[｜|]*tool\s*call\s*begin[｜|]*>(.*?)(?:<[｜|]*tool\s*call\s*end[｜|]*>|<[｜|]*tool\s*calls?\s*end[｜|]*>|$)"#).unwrap()
});

static DEEPSEEK_FN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(?:function|name|tool|action)\s*[:=]\s*['"]?([\w-]+)['"]?"#).unwrap()
});

static ATTRS_NAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)\b(?:name|function|tool|action)\s*=\s*['"]?([\w-]+)['"]?"#).unwrap()
});

static CHILD_TAG_NAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)<(?:name|tool_name|function_name|tool|function)>\s*([\w-]+)(?:</(?:name|tool_name|function_name|tool|function)>|\s|$)"#).unwrap()
});

static ARGUMENTS_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<(?:arguments|parameters)>(.*?)(?:</(?:arguments|parameters)>|$)"#).unwrap()
});

static DIRECT_TAG_TNAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)<(?:tool_name|function_name|tool)>([\w-]+)</(?:tool_name|function_name|tool)>")
        .unwrap()
});

static FENCE_CODEBLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```").unwrap());

static HEADLESS_COLON_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?is)(?:^|[\s`"'])(?P<delim>":|:|':)\s*["'](?P<tname>[\w-]+)["']\s*,\s*(?P<rest>.*)"#,
    )
    .unwrap()
});

static HEADLESS_KEY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)(?:^|[\s`"'])(?P<open>["'])?(?P<key>tool|name|function|action|tool_name|function_name)(?P<close>["'])?\s*:\s*(?P<rest>.*)"#).unwrap()
});

fn extract_direct_tags(text: &str, require_closing: bool) -> Vec<(String, String)> {
    static TAG_START_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"<([a-zA-Z0-9_-]+)>"#).unwrap());

    let mut results = Vec::new();
    let matches: Vec<(usize, usize, String)> = TAG_START_RE
        .captures_iter(text)
        .map(|caps| {
            let m = caps.get(0).unwrap();
            let tag = caps.get(1).unwrap().as_str().to_lowercase();
            (m.start(), m.end(), tag)
        })
        .collect();

    for i in 0..matches.len() {
        let (_, val_start, ref tag) = matches[i];
        if tag.starts_with('/') || tag.starts_with('!') || tag.starts_with('?') {
            continue;
        }
        let close_tag = format!("</{}>", tag);
        let next_tag_start = if i + 1 < matches.len() {
            matches[i + 1].0
        } else {
            text.len()
        };

        let slice = &text[val_start..];
        if let Some(pos) = slice.find(&close_tag) {
            results.push((tag.clone(), slice[..pos].to_string()));
        } else if !require_closing {
            let end_pos = (next_tag_start - val_start).min(slice.len());
            results.push((tag.clone(), slice[..end_pos].to_string()));
        }
    }

    results
}

pub struct ToolParser {
    config: ToolParserConfig,
}

impl Default for ToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolParser {
    pub fn new() -> Self {
        Self {
            config: ToolParserConfig::default(),
        }
    }

    pub fn with_config(config: ToolParserConfig) -> Self {
        Self { config }
    }

    pub fn normalize_name(&self, name: &str) -> String {
        let n = name.trim().to_lowercase();
        self.config.tool_aliases.get(&n).cloned().unwrap_or(n)
    }

    pub fn normalize_args(&self, name: &str, mut args: Map<String, Value>) -> Map<String, Value> {
        // 1. Unpack inner wrappers like "arguments", "parameters", "input"
        for wrap_key in &[
            "arguments",
            "parameters",
            "params",
            "args",
            "action_input",
            "tool_input",
            "input",
        ] {
            if let Some(sub) = args.remove(*wrap_key) {
                let sub_obj = match sub {
                    Value::String(s) => safe_json_loads(&s).and_then(|v| v.as_object().cloned()),
                    Value::Object(m) => Some(m),
                    _ => None,
                };
                if let Some(inner_map) = sub_obj {
                    for (k, v) in inner_map {
                        args.entry(k).or_insert(v);
                    }
                }
            }
        }

        // 2. Map positional argument keys like arg0, arg1
        if ["bash", "sh", "shell", "terminal"].contains(&name) {
            if let Some(arg0) = args.remove("arg0") {
                args.entry("command".to_string()).or_insert(arg0);
            }
        } else if ["search", "grep"].contains(&name) {
            if let Some(arg0) = args.remove("arg0") {
                args.entry("pattern".to_string()).or_insert(arg0);
            }
        } else if ["read_file", "view"].contains(&name) {
            if let Some(arg0) = args.remove("arg0") {
                args.entry("path".to_string()).or_insert(arg0);
            }
        } else if name == "write_file" {
            if let Some(arg0) = args.remove("arg0") {
                args.entry("path".to_string()).or_insert(arg0);
            }
            if let Some(arg1) = args.remove("arg1") {
                args.entry("content".to_string()).or_insert(arg1);
            }
        }

        // 3. Apply parameter aliases
        let mut final_args = Map::new();
        for (k, v) in args {
            let k_lower = k.to_lowercase();
            let norm_k = self
                .config
                .param_aliases
                .get(&k_lower)
                .cloned()
                .unwrap_or(k);
            final_args.insert(norm_k, v);
        }

        // 4. Convert numeric string parameters to int where expected
        for int_key in &["offset", "limit", "line_number", "line", "timeout"] {
            if let Some(Value::String(s)) = final_args.get(*int_key) {
                if let Ok(num) = s.trim().parse::<i64>() {
                    final_args.insert(int_key.to_string(), Value::Number(num.into()));
                }
            }
        }

        final_args
    }

    fn is_tool_allowed(&self, name: &str) -> bool {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return false;
        }
        match &self.config.allowed_tools {
            Some(set) => set.contains(trimmed),
            None => true,
        }
    }

    fn extract_name_and_args_from_value(
        &self,
        data: &Value,
    ) -> (Option<String>, Map<String, Value>) {
        if let Some(arr) = data.as_array() {
            if let Some(first) = arr.first() {
                return self.extract_name_and_args_from_value(first);
            }
            return (None, Map::new());
        }

        let Some(obj) = data.as_object() else {
            return (None, Map::new());
        };

        // 1. OpenAI tool_calls wrapper
        if let Some(tc) = obj
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
        {
            return self.extract_name_and_args_from_value(tc);
        }

        // 2. OpenAI function_call wrapper
        if let Some(fc) = obj.get("function_call").and_then(|v| v.as_object()) {
            return self.extract_name_and_args_from_value(&Value::Object(fc.clone()));
        }

        // 3. OpenAI function object
        if let Some(f) = obj.get("function").and_then(|v| v.as_object()) {
            return self.extract_name_and_args_from_value(&Value::Object(f.clone()));
        }

        let mut raw_name = obj
            .get("name")
            .or_else(|| obj.get("tool"))
            .or_else(|| obj.get("tool_name"))
            .or_else(|| obj.get("action"))
            .or_else(|| obj.get("function"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if let Some(ref n) = raw_name {
            if n.trim().is_empty() {
                raw_name = None;
            }
        }

        let mut args = Map::new();
        for k in &[
            "arguments",
            "parameters",
            "params",
            "args",
            "action_input",
            "tool_input",
        ] {
            if let Some(val) = obj.get(*k) {
                match val {
                    Value::String(s) => {
                        if let Some(parsed) = safe_json_loads(s) {
                            if let Some(m) = parsed.as_object() {
                                args = m.clone();
                            } else {
                                args.insert("input".to_string(), parsed);
                            }
                        } else {
                            args.insert("input".to_string(), Value::String(s.clone()));
                        }
                    }
                    Value::Object(m) => {
                        args = m.clone();
                    }
                    other => {
                        args.insert("input".to_string(), other.clone());
                    }
                }
                break;
            }
        }

        // Collect extra top-level keys
        let reserved = [
            "name",
            "tool",
            "tool_name",
            "action",
            "function",
            "type",
            "id",
            "arguments",
            "parameters",
            "params",
            "args",
            "action_input",
            "tool_input",
        ];
        for (k, v) in obj {
            if !reserved.contains(&k.as_str()) {
                args.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }

        (raw_name, args)
    }

    fn extract_calls_from_value(&self, data: &Value, source: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();

        if let Some(arr) = data.as_array() {
            for item in arr {
                calls.extend(self.extract_calls_from_value(item, source));
            }
            return calls;
        }

        let Some(obj) = data.as_object() else {
            return calls;
        };

        if let Some(tc_arr) = obj.get("tool_calls").and_then(|v| v.as_array()) {
            for item in tc_arr {
                calls.extend(self.extract_calls_from_value(item, source));
            }
            return calls;
        }

        if let Some(fc) = obj.get("function_call").and_then(|v| v.as_object()) {
            return self.extract_calls_from_value(&Value::Object(fc.clone()), source);
        }

        let (raw_name, args) = self.extract_name_and_args_from_value(data);
        if let Some(rn) = raw_name {
            let name = self.normalize_name(&rn);
            if self.is_tool_allowed(&name) {
                let norm_args = self.normalize_args(&name, args);
                calls.push(ToolCall::new(name, Value::Object(norm_args), source));
            }
        }

        calls
    }

    pub fn parse_calls(&self, raw_text: &str) -> Result<Vec<ToolCall>, ToolError> {
        let text = if self.config.strip_thinking {
            strip_thinking(raw_text)
        } else {
            raw_text.to_string()
        };

        // 1. DeepSeek native tokens
        let ds_calls = self.try_parse_deepseek_native(&text);
        if !ds_calls.is_empty() {
            return Ok(ds_calls);
        }

        // 2. XML / DSML / Claude
        let xml_calls = self.try_parse_xml_or_dsml(&text);
        if !xml_calls.is_empty() {
            return Ok(xml_calls);
        }

        // 3. Qwen agent tokens
        let qwen_calls = self.try_parse_qwen_agent(&text);
        if !qwen_calls.is_empty() {
            return Ok(qwen_calls);
        }

        // 4. ChatGLM
        let glm_calls = self.try_parse_chatglm(&text);
        if !glm_calls.is_empty() {
            return Ok(glm_calls);
        }

        // 5. Mistral [TOOL_CALLS]
        let mistral_calls = self.try_parse_mistral(&text);
        if !mistral_calls.is_empty() {
            return Ok(mistral_calls);
        }

        // 6. Llama 3.1 Python tag
        let llama_calls = self.try_parse_llama_python_tag(&text);
        if !llama_calls.is_empty() {
            return Ok(llama_calls);
        }

        // 7. Markdown JSON or plain JSON
        let json_calls = self.try_parse_json(&text);
        if !json_calls.is_empty() {
            return Ok(json_calls);
        }

        // 8. Python call expressions: func(arg="val")
        let py_calls = self.try_parse_python_expr(&text);
        if !py_calls.is_empty() {
            return Ok(py_calls);
        }

        // 9. ReAct
        let react_calls = self.try_parse_react(&text);
        if !react_calls.is_empty() {
            return Ok(react_calls);
        }

        // 10. Shell fallback
        let shell_calls = self.try_parse_shell(&text);
        if !shell_calls.is_empty() {
            return Ok(shell_calls);
        }

        Err(ToolError::new(format!(
            "No valid tool call found in response: {:?}",
            if text.len() > 100 {
                &text[..100]
            } else {
                &text
            }
        )))
    }

    pub fn parse(&self, text: &str) -> Result<ToolCall, ToolError> {
        let calls = self.parse_calls(text)?;
        calls
            .into_iter()
            .next()
            .ok_or_else(|| ToolError::new("Empty tool calls"))
    }

    pub fn try_parse(&self, text: &str) -> Option<ToolCall> {
        self.parse(text).ok()
    }

    pub fn try_parse_calls(&self, text: &str) -> Vec<ToolCall> {
        self.parse_calls(text).unwrap_or_default()
    }

    // --- Specific Parsers ---

    fn try_parse_deepseek_native(&self, text: &str) -> Vec<ToolCall> {
        if !text.contains("tool") && !text.contains("TOOL") {
            return Vec::new();
        }
        let lower = text.to_lowercase();
        if !lower.contains("tool call begin")
            && !lower.contains("tool_call_begin")
            && !lower.contains("tool sep")
            && !lower.contains("tool_sep")
        {
            return Vec::new();
        }

        let mut calls = Vec::new();
        let matches: Vec<_> = DEEPSEEK_NATIVE_BLOCK_RE.captures_iter(text).collect();
        let bodies: Vec<String> = if !matches.is_empty() {
            matches
                .iter()
                .map(|c| c.get(1).map_or("", |m| m.as_str()).trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        } else if lower.contains("tool sep") || lower.contains("tool_sep") {
            vec![text.trim().to_string()]
        } else {
            Vec::new()
        };

        for body in bodies {
            let mut tool_name = None;
            let mut json_str = body.clone();

            if let Some(pos) = body
                .find("<｜tool sep｜>")
                .or_else(|| body.find("<|tool sep|>"))
            {
                let header = body[..pos].trim();
                json_str = body[pos + "<｜tool sep｜>".len()..].trim().to_string();

                if let Some(caps) = DEEPSEEK_FN_RE.captures(header) {
                    tool_name = caps.get(1).map(|m| m.as_str().to_string());
                } else if !header.is_empty() && header != "type=function" {
                    tool_name = Some(header.to_string());
                }
            }

            if let Some(caps) = JSON_BLOCK_RE.captures(&json_str) {
                json_str = caps.get(1).map_or("", |m| m.as_str()).to_string();
            } else {
                let cands = extract_json_objects(&json_str);
                if let Some(first) = cands.first() {
                    json_str = first.clone();
                }
            }

            let mut args = Map::new();
            if let Some(Value::Object(m)) = safe_json_loads(&json_str) {
                let (d_name, d_args) = self.extract_name_and_args_from_value(&Value::Object(m));
                if tool_name.is_none() && d_name.is_some() {
                    tool_name = d_name;
                }
                args = d_args;
            }

            if let Some(tn) = tool_name {
                let norm_name = self.normalize_name(&tn);
                if self.is_tool_allowed(&norm_name) {
                    let norm_args = self.normalize_args(&norm_name, args);
                    calls.push(ToolCall::new(norm_name, Value::Object(norm_args), &body));
                }
            }
        }

        calls
    }

    fn try_parse_xml_or_dsml(&self, text: &str) -> Vec<ToolCall> {
        if !text.contains('<') {
            return Vec::new();
        }

        let mut calls = Vec::new();

        // 1. Direct tag format: <tool_name>search</tool_name><pattern>...</pattern>
        let has_invoke_wrapper = INVOKE_RE.is_match(text);
        let lower = text.to_lowercase();
        if !has_invoke_wrapper
            && (lower.contains("<tool_name>")
                || lower.contains("<function_name>")
                || lower.contains("<tool>"))
        {
            if let Some(caps) = DIRECT_TAG_TNAME_RE.captures(text) {
                let raw_tname = caps.get(1).unwrap().as_str();
                let tname = self.normalize_name(raw_tname);
                let mut params = Map::new();

                for (ptag, pval) in extract_direct_tags(text, true) {
                    if !["tool_name", "function_name", "tool"].contains(&ptag.as_str()) {
                        let is_code = self.config.code_param_keys.contains(&ptag);
                        let val = clean_param_val(&pval, is_code);
                        params.insert(ptag, Value::String(val));
                    }
                }

                if self.is_tool_allowed(&tname) {
                    let norm_args = self.normalize_args(&tname, params);
                    calls.push(ToolCall::new(tname, Value::Object(norm_args), text));
                    return calls;
                }
            }
        }

        // 2. Parse <invoke> / <tool_call> / <dsml:tool> tags
        for m in INVOKE_RE.captures_iter(text) {
            let colon_tool = m.name("colon_tool").map(|c| c.as_str().to_string());
            let attrs = m.name("attrs").map_or("", |a| a.as_str());
            let body = m.name("body").map_or("", |b| b.as_str());

            let mut tool_name = colon_tool;

            if tool_name.is_none() {
                if let Some(caps) = ATTRS_NAME_RE.captures(attrs) {
                    tool_name = caps.get(1).map(|m| m.as_str().to_string());
                }
            }

            if tool_name.is_none() {
                if let Some(caps) = CHILD_TAG_NAME_RE.captures(body) {
                    tool_name = caps.get(1).map(|m| m.as_str().to_string());
                }
            }

            let mut params = Map::new();

            // Extract <parameter name="...">
            for (pname, pval) in extract_xml_parameters(body) {
                let is_code = self.config.code_param_keys.contains(&pname);
                params.insert(pname, Value::String(clean_param_val(&pval, is_code)));
            }

            // Extract <arguments>...</arguments>
            if let Some(caps) = ARGUMENTS_BLOCK_RE.captures(body) {
                let wrap_body = caps.get(1).map_or("", |m| m.as_str()).trim();
                let cands = extract_json_objects(wrap_body);
                if let Some(first) = cands.first() {
                    if let Some(Value::Object(m)) = safe_json_loads(first) {
                        for (k, v) in m {
                            params.insert(k, v);
                        }
                    }
                }
                if params.is_empty() {
                    for (ptag, pval) in extract_direct_tags(wrap_body, false) {
                        if !["arguments", "parameters"].contains(&ptag.as_str()) {
                            let is_code = self.config.code_param_keys.contains(&ptag);
                            let val = clean_param_val(&pval, is_code);
                            params.insert(ptag, Value::String(val));
                        }
                    }
                }
            }

            // Extract direct child tags if still empty
            if params.is_empty() {
                for (ptag, pval) in extract_direct_tags(body, true) {
                    if ![
                        "name",
                        "tool_name",
                        "function_name",
                        "tool",
                        "function",
                        "arguments",
                        "parameters",
                    ]
                    .contains(&ptag.as_str())
                    {
                        let is_code = self.config.code_param_keys.contains(&ptag);
                        let val = clean_param_val(&pval, is_code);
                        params.insert(ptag, Value::String(val));
                    }
                }
            }

            if tool_name.is_none() {
                if let Some(Value::String(s)) = params
                    .remove("tool")
                    .or_else(|| params.remove("name"))
                    .or_else(|| params.remove("action"))
                {
                    tool_name = Some(s);
                }
            }

            // Embedded JSON in body
            if params.is_empty() {
                let mut json_str = None;
                if let Some(caps) = JSON_BLOCK_RE.captures(body) {
                    json_str = caps.get(1).map(|m| m.as_str().to_string());
                } else {
                    let cands = extract_json_objects(body);
                    if let Some(first) = cands.first() {
                        json_str = Some(first.clone());
                    }
                }
                if let Some(js) = json_str {
                    if let Some(Value::Object(m)) = safe_json_loads(&js) {
                        let (d_name, d_args) =
                            self.extract_name_and_args_from_value(&Value::Object(m));
                        if tool_name.is_none() && d_name.is_some() {
                            tool_name = d_name;
                        }
                        for (k, v) in d_args {
                            params.insert(k, v);
                        }
                    }
                }
            }

            if let Some(tn) = tool_name {
                let norm_name = self.normalize_name(&tn);
                if self.is_tool_allowed(&norm_name) {
                    let norm_args = self.normalize_args(&norm_name, params);
                    let raw_src = m.get(0).map_or("", |c| c.as_str());
                    calls.push(ToolCall::new(norm_name, Value::Object(norm_args), raw_src));
                }
            }
        }

        if !calls.is_empty() {
            return calls;
        }

        // 3. Standalone parameters directly inside <tool_calls> without invoke wrapper
        if lower.contains("dsml")
            || lower.contains("tool_calls")
            || lower.contains("function_calls")
        {
            let mut standalone_params = Map::new();
            for (pname, pval) in extract_xml_parameters(text) {
                let is_code = self.config.code_param_keys.contains(&pname);
                standalone_params.insert(pname, Value::String(clean_param_val(&pval, is_code)));
            }

            let mut tool_name = standalone_params
                .remove("tool")
                .or_else(|| standalone_params.remove("name"))
                .or_else(|| standalone_params.remove("action"))
                .and_then(|v| v.as_str().map(|s| s.to_string()));

            if tool_name.is_none() {
                if let Some(set) = &self.config.allowed_tools {
                    for t_tag in set {
                        if !lower.contains(&t_tag.to_lowercase()) {
                            continue;
                        }
                        let re = Regex::new(&format!(
                            r#"(?i)<[｜|]*(?:dsml[｜|]*)?{}\b[^>]*>"#,
                            regex::escape(t_tag)
                        ))
                        .unwrap();
                        if re.is_match(text) {
                            tool_name = Some(t_tag.clone());
                            break;
                        }
                    }
                }
            }

            if let Some(tn) = tool_name {
                let norm_name = self.normalize_name(&tn);
                if self.is_tool_allowed(&norm_name) {
                    let norm_args = self.normalize_args(&norm_name, standalone_params);
                    calls.push(ToolCall::new(norm_name, Value::Object(norm_args), text));
                }
            }
        }

        calls
    }

    fn try_parse_qwen_agent(&self, text: &str) -> Vec<ToolCall> {
        if !text.contains("action_name") {
            return Vec::new();
        }

        let mut calls = Vec::new();
        for m in QWEN_AGENT_RE.captures_iter(text) {
            let raw_name = m.name("name").unwrap().as_str().trim();
            let name = self.normalize_name(raw_name);
            if !self.is_tool_allowed(&name) {
                continue;
            }

            let args_str = m.name("args").map_or("", |m| m.as_str()).trim();
            let mut args = Map::new();
            if !args_str.is_empty() {
                if let Some(Value::Object(map)) = safe_json_loads(args_str) {
                    args = map;
                } else if let Some(v) = safe_json_loads(args_str) {
                    args.insert("input".to_string(), v);
                } else {
                    args.insert("input".to_string(), Value::String(args_str.to_string()));
                }
            }

            let norm_args = self.normalize_args(&name, args);
            let raw_src = m.get(0).map_or("", |m| m.as_str());
            calls.push(ToolCall::new(name, Value::Object(norm_args), raw_src));
        }

        calls
    }

    fn try_parse_chatglm(&self, text: &str) -> Vec<ToolCall> {
        if !text.contains("FUNCTION") {
            return Vec::new();
        }

        let mut calls = Vec::new();
        for m in CHATGLM_RE.captures_iter(text) {
            let raw_name = m.name("name").unwrap().as_str().trim();
            let name = self.normalize_name(raw_name);
            if !self.is_tool_allowed(&name) {
                continue;
            }

            let args_str = m.name("args").map_or("", |m| m.as_str()).trim();
            let mut args = Map::new();
            if !args_str.is_empty() {
                if let Some(Value::Object(map)) = safe_json_loads(args_str) {
                    args = map;
                } else if let Some(v) = safe_json_loads(args_str) {
                    args.insert("input".to_string(), v);
                } else {
                    args.insert("input".to_string(), Value::String(args_str.to_string()));
                }
            }

            let norm_args = self.normalize_args(&name, args);
            let raw_src = m.get(0).map_or("", |m| m.as_str());
            calls.push(ToolCall::new(name, Value::Object(norm_args), raw_src));
        }

        calls
    }

    fn try_parse_mistral(&self, text: &str) -> Vec<ToolCall> {
        if !text.contains("[TOOL_CALL") {
            return Vec::new();
        }

        let mut calls = Vec::new();
        for m in MISTRAL_TOOL_CALL_RE.captures_iter(text) {
            let body = m.name("body").unwrap().as_str().trim();
            let raw_src = m.get(0).unwrap().as_str();
            if let Some(v) = safe_json_loads(body) {
                calls.extend(self.extract_calls_from_value(&v, raw_src));
            }
        }

        calls
    }

    fn try_parse_llama_python_tag(&self, text: &str) -> Vec<ToolCall> {
        if !text.contains("python_tag") {
            return Vec::new();
        }

        let mut calls = Vec::new();
        for m in LLAMA_PYTHON_TAG_RE.captures_iter(text) {
            let body = m.name("body").unwrap().as_str().trim();
            let raw_src = m.get(0).unwrap().as_str();

            // 1. Try parsing body as JSON
            let sub_calls = self.try_parse_json(body);
            if !sub_calls.is_empty() {
                for mut c in sub_calls {
                    c.raw_source = raw_src.to_string();
                    calls.push(c);
                }
                continue;
            }

            // 2. Try parsing body as Python call
            let sub_calls = self.try_parse_python_expr(body);
            if !sub_calls.is_empty() {
                for mut c in sub_calls {
                    c.raw_source = raw_src.to_string();
                    calls.push(c);
                }
            }
        }

        calls
    }

    fn repair_headless_json(&self, raw: &str) -> Vec<String> {
        let mut cands = Vec::new();

        // 1. Colon prefix omission: `: "tool_name", ...` or `": "tool_name", ...`
        if let Some(caps) = HEADLESS_COLON_RE.captures(raw) {
            let tname = caps.name("tname").unwrap().as_str();
            let rest = caps.name("rest").unwrap().as_str();
            cands.push(format!(r#"{{"tool": "{}", {}"#, tname, rest));
            cands.push(format!(r#"{{"name": "{}", {}"#, tname, rest));
        }

        // 2. Unquoted or unbraced key omission: `tool": "bash", ...` or `"tool": "bash", ...`
        if let Some(caps) = HEADLESS_KEY_RE.captures(raw) {
            let open_q = caps.name("open");
            let close_q = caps.name("close").map_or("\"", |m| m.as_str());
            let key = caps.name("key").unwrap().as_str();
            let rest = caps.name("rest").unwrap().as_str();

            if let Some(oq) = open_q {
                let start_idx = oq.start();
                cands.push(format!("{{{}", &raw[start_idx..]));
            } else {
                cands.push(format!("{{{}{}{}: {}", close_q, key, close_q, rest));
            }
        }

        cands
    }

    fn try_parse_json(&self, raw: &str) -> Vec<ToolCall> {
        if !raw.contains('{') && !raw.contains('[') && !raw.contains("```") && !raw.contains(':') {
            return Vec::new();
        }

        let mut calls = Vec::new();

        // =====================================================================
        // TIER 1: STRICT SPECIFICATION COMPLIANCE (Fast-Path)
        // Checks canonical markdown JSON code blocks, RFC-8259 arrays, and balanced objects.
        // =====================================================================

        // 1. Markdown code blocks ```json ... ```
        for mb in JSON_BLOCK_RE.captures_iter(raw) {
            let blob = mb.get(1).unwrap().as_str();
            if let Some(v) = safe_json_loads(blob) {
                calls.extend(self.extract_calls_from_value(&v, blob));
            }
        }
        for mb in JSON_ARRAY_BLOCK_RE.captures_iter(raw) {
            let blob = mb.get(1).unwrap().as_str();
            if let Some(v) = safe_json_loads(blob) {
                calls.extend(self.extract_calls_from_value(&v, blob));
            }
        }
        if !calls.is_empty() {
            return calls;
        }

        // 2. Top-level JSON array
        let trimmed = raw.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            if let Some(v) = safe_json_loads(trimmed) {
                let extracted = self.extract_calls_from_value(&v, trimmed);
                if !extracted.is_empty() {
                    return extracted;
                }
            }
        }

        // 3. Balanced JSON objects via standard bracket tracking
        let candidates = extract_json_objects(raw);
        for cand in candidates {
            if let Some(v) = safe_json_loads(&cand) {
                calls.extend(self.extract_calls_from_value(&v, &cand));
            }
        }
        if !calls.is_empty() {
            return calls;
        }

        // =====================================================================
        // TIER 2: FAULT-TOLERANT RECOVERY (ParseGuard Layer)
        // If input deviates from balanced JSON specification (e.g. MiniMax / prompt-prefill
        // prefix omissions: `tool": "bash", ...` or `: "read_file", ...`), attempt deterministic recovery.
        // =====================================================================
        let recovered = self.repair_headless_json(raw);
        for cand in recovered {
            let sub_candidates = extract_json_objects(&cand);
            for sub_cand in sub_candidates {
                if let Some(v) = safe_json_loads(&sub_cand) {
                    calls.extend(self.extract_calls_from_value(&v, &sub_cand));
                }
            }
            if !calls.is_empty() {
                return calls;
            }
        }

        calls
    }

    fn try_parse_python_expr(&self, text: &str) -> Vec<ToolCall> {
        if !text.contains('(') {
            return Vec::new();
        }

        let parsed = crate::python_calls::extract_python_function_calls(text, &self.config);

        if !parsed.is_empty() {
            return parsed
                .into_iter()
                .map(|p| {
                    let norm_args = self.normalize_args(&p.name, p.args);
                    ToolCall::new(p.name, Value::Object(norm_args), p.raw_source)
                })
                .collect();
        }

        Vec::new()
    }

    fn try_parse_react(&self, text: &str) -> Vec<ToolCall> {
        let lower = text.to_lowercase();
        if !lower.contains("action:") {
            return Vec::new();
        }

        let mut calls = Vec::new();
        let matches: Vec<_> = REACT_HEADER_RE.captures_iter(text).collect();
        for i in 0..matches.len() {
            let caps = &matches[i];
            let raw_name = caps.name("name").unwrap().as_str().trim();
            let name = self.normalize_name(raw_name);
            if !self.is_tool_allowed(&name) {
                continue;
            }

            let arg_start = caps.get(0).unwrap().end();
            let upper_bound = if i + 1 < matches.len() {
                matches[i + 1].get(0).unwrap().start()
            } else {
                text.len()
            };

            let mut action_input = text[arg_start..upper_bound].trim();
            for stop in &["\nAction:", "\naction:", "\nThought:", "\nthought:"] {
                if let Some(pos) = action_input.find(stop) {
                    action_input = action_input[..pos].trim();
                }
            }

            if action_input.starts_with("```") {
                if let Some(c) = FENCE_CODEBLOCK_RE.captures(action_input) {
                    action_input = c.get(1).unwrap().as_str().trim();
                }
            }

            let mut args = Map::new();
            let cands = extract_json_objects(action_input);
            if let Some(first) = cands.first() {
                if let Some(Value::Object(map)) = safe_json_loads(first) {
                    args = map;
                }
            }

            if args.is_empty() {
                if name == "bash" {
                    args.insert(
                        "command".to_string(),
                        Value::String(action_input.to_string()),
                    );
                } else {
                    args.insert("input".to_string(), Value::String(action_input.to_string()));
                }
            }

            let norm_args = self.normalize_args(&name, args);
            let raw_src = &text[caps.get(0).unwrap().start()..upper_bound];
            calls.push(ToolCall::new(
                name,
                Value::Object(norm_args),
                raw_src.trim(),
            ));
        }

        calls
    }

    fn try_parse_shell(&self, raw: &str) -> Vec<ToolCall> {
        if !self.config.allow_shell_fallback {
            return Vec::new();
        }

        let mut calls = Vec::new();
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("$ ") || trimmed.starts_with("# ") {
                let cmd = trimmed[2..].trim();
                let mut args = Map::new();
                args.insert("command".to_string(), Value::String(cmd.to_string()));
                calls.push(ToolCall::new("bash", Value::Object(args), trimmed));
            }
        }

        calls
    }
}

/// Parses a single tool call from text using default settings.
pub fn parse_tool_call(text: &str) -> Result<ToolCall, ToolError> {
    ToolParser::new().parse(text)
}

/// Parses all tool calls from text using default settings.
pub fn parse_tool_calls(text: &str) -> Result<Vec<ToolCall>, ToolError> {
    ToolParser::new().parse_calls(text)
}

/// Non-raising variant of parse_tool_call. Returns None on failure.
pub fn try_parse_tool_call(text: &str) -> Option<ToolCall> {
    parse_tool_call(text).ok()
}

/// Non-raising variant of parse_tool_calls. Returns an empty Vec on failure.
pub fn try_parse_tool_calls(text: &str) -> Vec<ToolCall> {
    parse_tool_calls(text).unwrap_or_default()
}
