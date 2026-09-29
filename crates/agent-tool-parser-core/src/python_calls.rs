// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

//! High-performance extractor for Python function call expressions in markdown and raw agent outputs.
//! Supports multiline argument blocks, triple-quoted strings (`"""` / `'''`),
//! domestic LLM dialects (GigaChat, YandexGPT), and Russian semantic action prefixes.

use crate::cleaners::safe_json_loads;
use crate::models::{ToolCall, ToolParserConfig};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

static CODEBLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)```(?:python|py)?\s*(.*?)\s*```").unwrap());

static PYTHON_BUILTINS: &[&str] = &[
    "print",
    "len",
    "range",
    "dict",
    "list",
    "str",
    "int",
    "float",
    "bool",
    "set",
    "tuple",
    "isinstance",
    "type",
    "enumerate",
    "zip",
    "sum",
    "min",
    "max",
    "open",
    "help",
    "id",
    "input",
    "eval",
    "exec",
    "compile",
];

static PYTHON_KEYWORDS: &[&str] = &[
    "def", "class", "return", "yield", "raise", "if", "elif", "else", "while", "for", "try",
    "except", "with", "async", "lambda",
];

static PREFIXES: &[&str] = &[
    "вызов функции:",
    "вызов инструмента:",
    "действие:",
    "вызов:",
    "инструмент:",
    "функция:",
    "action:",
    "call:",
    "tool:",
    "function:",
];

/// An extracted raw Python function call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedPythonCall {
    pub name: String,
    pub args: Map<String, Value>,
    pub raw_args: String,
    pub raw_source: String,
}

impl ParsedPythonCall {
    pub fn to_tool_call(self) -> ToolCall {
        ToolCall::new(self.name, Value::Object(self.args), self.raw_source)
    }
}

/// Unescapes string escape sequences (`\n`, `\t`, `\"`, `\'`, `\\`).
pub fn unescape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('\'') => out.push('\''),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Parses a single Python argument value (string, triple-quoted string, JSON, number, bool, null).
pub fn parse_python_arg_value(val_str: &str) -> Value {
    let trimmed = val_str.trim();
    if trimmed.is_empty() {
        return Value::String(String::new());
    }

    // 1. Triple-quoted strings """...""" or '''...'''
    if (trimmed.starts_with("\"\"\"") && trimmed.ends_with("\"\"\"") && trimmed.len() >= 6)
        || (trimmed.starts_with("'''") && trimmed.ends_with("'''") && trimmed.len() >= 6)
    {
        let inner = &trimmed[3..trimmed.len() - 3];
        return Value::String(unescape_string(inner));
    }

    // 2. Double-quoted strings "..."
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        let inner = &trimmed[1..trimmed.len() - 1];
        return Value::String(unescape_string(inner));
    }

    // 3. Single-quoted strings '...'
    if trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2 {
        let inner = &trimmed[1..trimmed.len() - 1];
        return Value::String(unescape_string(inner));
    }

    // 4. Booleans and None
    match trimmed {
        "True" | "true" => return Value::Bool(true),
        "False" | "false" => return Value::Bool(false),
        "None" | "null" => return Value::Null,
        _ => {}
    }

    // 5. JSON dicts or arrays
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        if let Some(val) = safe_json_loads(trimmed) {
            return val;
        }
    }

    // 6. Numeric literals
    if let Ok(i) = trimmed.parse::<i64>() {
        return Value::Number(serde_json::Number::from(i));
    }
    if let Ok(f) = trimmed.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Value::Number(n);
        }
    }

    Value::String(trimmed.to_string())
}

/// Splits arguments at top-level commas, respecting single/double/triple quotes and nested braces.
pub fn split_python_args(raw_args: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = raw_args.chars().collect();
    let mut i = 0;
    let n = chars.len();

    let mut in_triple_double = false;
    let mut in_triple_single = false;
    let mut in_double = false;
    let mut in_single = false;
    let mut escape = false;
    let mut paren_depth = 0;
    let mut bracket_depth = 0;
    let mut brace_depth = 0;

    while i < n {
        let c = chars[i];

        if escape {
            cur.push(c);
            escape = false;
            i += 1;
            continue;
        }

        if c == '\\' && (in_triple_double || in_triple_single || in_double || in_single) {
            cur.push(c);
            escape = true;
            i += 1;
            continue;
        }

        // Check triple quotes
        if !in_single && !in_double {
            if !in_triple_single
                && i + 2 < n
                && chars[i] == '"'
                && chars[i + 1] == '"'
                && chars[i + 2] == '"'
            {
                in_triple_double = !in_triple_double;
                cur.push_str("\"\"\"");
                i += 3;
                continue;
            }
            if !in_triple_double
                && i + 2 < n
                && chars[i] == '\''
                && chars[i + 1] == '\''
                && chars[i + 2] == '\''
            {
                in_triple_single = !in_triple_single;
                cur.push_str("'''");
                i += 3;
                continue;
            }
        }

        if !in_triple_double && !in_triple_single {
            if c == '"' && !in_single {
                in_double = !in_double;
                cur.push(c);
                i += 1;
                continue;
            }
            if c == '\'' && !in_double {
                in_single = !in_single;
                cur.push(c);
                i += 1;
                continue;
            }
        }

        if !in_triple_double && !in_triple_single && !in_double && !in_single {
            match c {
                '(' => paren_depth += 1,
                ')' => {
                    if paren_depth > 0 {
                        paren_depth -= 1;
                    }
                }
                '[' => bracket_depth += 1,
                ']' => {
                    if bracket_depth > 0 {
                        bracket_depth -= 1;
                    }
                }
                '{' => brace_depth += 1,
                '}' => {
                    if brace_depth > 0 {
                        brace_depth -= 1;
                    }
                }
                ',' if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                    tokens.push(cur.trim().to_string());
                    cur.clear();
                    i += 1;
                    continue;
                }
                _ => {}
            }
        }

        cur.push(c);
        i += 1;
    }

    let trimmed = cur.trim();
    if !trimmed.is_empty() {
        tokens.push(trimmed.to_string());
    }

    tokens
}

/// Parses a raw Python function arguments string into a key-value Map.
pub fn parse_python_kwargs(raw_args: &str) -> Map<String, Value> {
    let mut map = Map::new();
    let tokens = split_python_args(raw_args);
    let mut pos_idx = 0;

    for tok in tokens {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }

        // Look for unquoted '=' separating kwarg name and value
        let mut eq_pos: Option<usize> = None;
        let chars: Vec<char> = tok.chars().collect();
        let mut in_str = false;
        let mut str_q = ' ';
        let mut esc = false;
        let mut depth = 0;

        for (idx, &ch) in chars.iter().enumerate() {
            if esc {
                esc = false;
                continue;
            }
            if ch == '\\' && in_str {
                esc = true;
                continue;
            }
            if (ch == '"' || ch == '\'') && !esc {
                if !in_str {
                    in_str = true;
                    str_q = ch;
                } else if str_q == ch {
                    in_str = false;
                }
            } else if !in_str {
                if ch == '(' || ch == '[' || ch == '{' {
                    depth += 1;
                } else if ch == ')' || ch == ']' || ch == '}' {
                    if depth > 0 {
                        depth -= 1;
                    }
                } else if ch == '=' && depth == 0 {
                    eq_pos = Some(idx);
                    break;
                }
            }
        }

        if let Some(pos) = eq_pos {
            let key: String = chars[..pos].iter().collect();
            let val: String = chars[pos + 1..].iter().collect();
            let key = key.trim().to_string();
            map.insert(key, parse_python_arg_value(&val));
        } else {
            let key = format!("arg{}", pos_idx);
            pos_idx += 1;
            map.insert(key, parse_python_arg_value(tok));
        }
    }

    map
}

/// Scans text for balanced Python function calls, supporting multiline arguments,
/// triple-quoted strings, and Russian action prefixes.
pub fn extract_python_function_calls(
    text: &str,
    config: &ToolParserConfig,
) -> Vec<ParsedPythonCall> {
    let mut calls = Vec::new();

    // Check code blocks first
    for caps in CODEBLOCK_RE.captures_iter(text) {
        let block = caps.get(1).unwrap().as_str();
        let sub = scan_calls_in_text(block, config);
        calls.extend(sub);
    }

    if !calls.is_empty() {
        return calls;
    }

    // Scan full text
    scan_calls_in_text(text, config)
}

fn scan_calls_in_text(text: &str, config: &ToolParserConfig) -> Vec<ParsedPythonCall> {
    let mut calls = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        if chars[i].1 == '(' {
            let open_paren = i;
            // Scan backwards to extract function name and prefix
            let mut p = open_paren;
            while p > 0 && chars[p - 1].1.is_whitespace() {
                p -= 1;
            }
            let name_end = p;
            while p > 0
                && (chars[p - 1].1.is_alphanumeric()
                    || chars[p - 1].1 == '_'
                    || chars[p - 1].1 == '.')
            {
                p -= 1;
            }
            let name_start = p;

            if name_end > name_start {
                let func_ident: String = chars[name_start..name_end]
                    .iter()
                    .map(|(_, c)| *c)
                    .collect();

                // Check what precedes name_start
                let mut prefix_start = name_start;
                while prefix_start > 0
                    && chars[prefix_start - 1].1 != '\n'
                    && chars[prefix_start - 1].1 != '\r'
                {
                    prefix_start -= 1;
                }
                let before_name: String = chars[prefix_start..name_start]
                    .iter()
                    .map(|(_, c)| *c)
                    .collect();
                let before_trimmed = before_name.trim();

                // Validate before_name: must be empty, whitespace, assignment rejection, or known prefix
                let mut is_prefix_valid =
                    before_trimmed.is_empty() || before_trimmed.ends_with('`');
                let before_lower = before_trimmed.to_lowercase();
                for &pref in PREFIXES {
                    if before_lower.ends_with(pref) {
                        is_prefix_valid = true;
                        break;
                    }
                }

                // If line contains '=' before the function call, it's an assignment like `result = ...`
                if before_trimmed.contains('=') && !is_prefix_valid {
                    i += 1;
                    continue;
                }

                if is_prefix_valid {
                    let func_name = normalize_raw_function_name(&func_ident);
                    let is_builtin = PYTHON_BUILTINS.contains(&func_name.as_str());
                    let is_keyword = PYTHON_KEYWORDS.contains(&func_name.as_str());

                    if !is_keyword && (config.allowed_tools.is_some() || !is_builtin) {
                        // Scan forward for balancing ')'
                        if let Some(close_paren) = find_matching_close_paren(&chars, open_paren) {
                            let raw_args_str: String = chars[open_paren + 1..close_paren]
                                .iter()
                                .map(|(_, c)| *c)
                                .collect();
                            let raw_source_str: String = chars[prefix_start..=close_paren]
                                .iter()
                                .map(|(_, c)| *c)
                                .collect();

                            let parsed_args = parse_python_kwargs(&raw_args_str);
                            let normalized_name = config
                                .tool_aliases
                                .get(&func_name.to_lowercase())
                                .cloned()
                                .unwrap_or_else(|| func_name.clone());

                            let is_allowed = config
                                .allowed_tools
                                .as_ref()
                                .is_none_or(|set| set.contains(&normalized_name));

                            if is_allowed {
                                calls.push(ParsedPythonCall {
                                    name: normalized_name,
                                    args: parsed_args,
                                    raw_args: raw_args_str,
                                    raw_source: raw_source_str.trim().to_string(),
                                });
                                i = close_paren + 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }
        i += 1;
    }

    calls
}

fn normalize_raw_function_name(raw: &str) -> String {
    let raw = raw.strip_prefix("call:").unwrap_or(raw).trim();
    if raw.contains('.') {
        let parts: Vec<&str> = raw.split('.').collect();
        if parts.len() >= 2 && ["call", "run", "execute", "invoke"].contains(&parts[1]) {
            parts[0].to_string()
        } else if parts.len() >= 2 && ["tool", "tools", "action", "functions"].contains(&parts[0]) {
            parts[1].to_string()
        } else {
            parts.last().unwrap_or(&raw).to_string()
        }
    } else {
        raw.to_string()
    }
}

fn find_matching_close_paren(chars: &[(usize, char)], open_idx: usize) -> Option<usize> {
    let mut depth = 0;
    let mut in_triple_double = false;
    let mut in_triple_single = false;
    let mut in_double = false;
    let mut in_single = false;
    let mut escape = false;

    let n = chars.len();
    let mut j = open_idx;

    while j < n {
        let (_, c) = chars[j];

        if escape {
            escape = false;
            j += 1;
            continue;
        }

        if c == '\\' && (in_triple_double || in_triple_single || in_double || in_single) {
            escape = true;
            j += 1;
            continue;
        }

        if !in_single && !in_double {
            if !in_triple_single
                && j + 2 < n
                && chars[j].1 == '"'
                && chars[j + 1].1 == '"'
                && chars[j + 2].1 == '"'
            {
                in_triple_double = !in_triple_double;
                j += 3;
                continue;
            }
            if !in_triple_double
                && j + 2 < n
                && chars[j].1 == '\''
                && chars[j + 1].1 == '\''
                && chars[j + 2].1 == '\''
            {
                in_triple_single = !in_triple_single;
                j += 3;
                continue;
            }
        }

        if !in_triple_double && !in_triple_single {
            if c == '"' && !in_single {
                in_double = !in_double;
                j += 1;
                continue;
            }
            if c == '\'' && !in_double {
                in_single = !in_single;
                j += 1;
                continue;
            }
        }

        if !in_triple_double && !in_triple_single && !in_double && !in_single {
            if c == '(' {
                depth += 1;
            } else if c == ')' {
                depth -= 1;
                if depth == 0 {
                    return Some(j);
                }
            }
        }

        j += 1;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiline_python_call_with_triple_quotes() {
        let text = r#"```python
str_replace(
    path="lib/matplotlib/dates.py",
    old_str="""def date2num(d):
    return d""",
    new_str="""def date2num(d):
    return _date2num(d)"""
)
```"#;
        let config = ToolParserConfig::default();
        let calls = extract_python_function_calls(text, &config);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "str_replace");
        assert_eq!(calls[0].args["path"], "lib/matplotlib/dates.py");
        assert_eq!(calls[0].args["old_str"], "def date2num(d):\n    return d");
        assert_eq!(
            calls[0].args["new_str"],
            "def date2num(d):\n    return _date2num(d)"
        );
    }

    #[test]
    fn test_russian_action_prefix() {
        let text = "Вызов функции: edit_file(path=\"src/models.py\", command=\"replace\")";
        let config = ToolParserConfig::default();
        let calls = extract_python_function_calls(text, &config);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "str_replace");
        assert_eq!(calls[0].args["path"], "src/models.py");
        assert_eq!(calls[0].args["command"], "replace");
    }

    #[test]
    fn test_bare_python_call() {
        let text =
            "Here is the tool call:\nwrite_file(path=\"solution.py\", content=\"# fix\\n\")\nDone.";
        let config = ToolParserConfig::default();
        let calls = extract_python_function_calls(text, &config);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(calls[0].args["path"], "solution.py");
        assert_eq!(calls[0].args["content"], "# fix\n");
    }
}
