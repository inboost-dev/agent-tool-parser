// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

use once_cell::sync::Lazy;
use regex::Regex;
use std::borrow::Cow;

static THINK_OPEN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)<[｜|]?(?:think|thought|reasoning)[｜|]?>").unwrap());

static THINK_CLOSE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)</[｜|]?(?:think|thought|reasoning)[｜|]?>").unwrap());

static TOOL_START_CANDIDATES: &[&str] = &[
    "<tool_call",
    "<tool_calls",
    "<tools",
    "<tool",
    "<invoke",
    "<dsml",
    "<call",
    "<tool_use",
    "<ant_tool_use",
    "<action_start",
    "<action_name",
    "<python_tag",
    "```",
    "action:",
    "✿function✿",
];

static CDATA_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)<!\[CDATA\[(.*?)(?:\]\]>|$)").unwrap());

static SINGLE_QUOTE_KEYS_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"([{,]\s*)'([^']+)'(\s*:)"#).unwrap());

static SINGLE_QUOTE_VALS_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(:\s*)'([^']*)'(\s*[,}])"#).unwrap());

static TRAILING_COMMA_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r",\s*([}\]])").unwrap());

static PY_TRUE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bTrue\b").unwrap());
static PY_FALSE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bFalse\b").unwrap());
static PY_NONE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bNone\b").unwrap());

static BRACE_COLON_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^\{\s*\"(?::|\s*:)\s*"#).unwrap());
static PREFIX_COLON_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^\s*\"(?::|\s*:)\s*"#).unwrap());
static RAW_COLON_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^\s*:\s*"#).unwrap());

/// Normalizes truncated JSON prefixes (e.g. from LLMs dropping `{"` or `{"tool`).
///
/// Returns `Cow::Borrowed` if no normalization is needed (zero-copy / zero allocation).
pub fn normalize_truncated_json_prefix<'a>(s: &'a str) -> Cow<'a, str> {
    let trimmed = s.trim_start();
    if trimmed.starts_with("tool\":")
        || trimmed.starts_with("name\":")
        || trimmed.starts_with("action\":")
    {
        let offset = s.len() - trimmed.len();
        Cow::Owned(format!("{}{{\"{}", &s[..offset], trimmed))
    } else if trimmed.starts_with("\"tool\":")
        || trimmed.starts_with("\"name\":")
        || trimmed.starts_with("\"action\":")
    {
        let offset = s.len() - trimmed.len();
        Cow::Owned(format!("{}{{{}", &s[..offset], trimmed))
    } else if let Some(m) = BRACE_COLON_RE.find(trimmed) {
        let offset = s.len() - trimmed.len();
        Cow::Owned(format!(
            "{}{{\"tool\": {}",
            &s[..offset],
            &trimmed[m.end()..]
        ))
    } else if let Some(m) = PREFIX_COLON_RE.find(trimmed) {
        let offset = s.len() - trimmed.len();
        Cow::Owned(format!(
            "{}{{\"tool\": {}",
            &s[..offset],
            &trimmed[m.end()..]
        ))
    } else if let Some(m) = RAW_COLON_RE.find(trimmed) {
        let offset = s.len() - trimmed.len();
        Cow::Owned(format!(
            "{}{{\"tool\": {}",
            &s[..offset],
            &trimmed[m.end()..]
        ))
    } else {
        Cow::Borrowed(s)
    }
}

/// Strips internal reasoning / thinking tags (<think>...</think>).
pub fn strip_thinking(text: &str) -> String {
    if !text.contains('<') {
        return text.to_string();
    }
    if let Some(open_m) = THINK_OPEN_RE.find(text) {
        let after_open = &text[open_m.end()..];
        // 1. Look for explicit closing tag
        if let Some(close_m) = THINK_CLOSE_RE.find(after_open) {
            let before = &text[..open_m.start()];
            let after = &after_open[close_m.end()..];
            let result = format!("{}{}", before, after);
            if THINK_OPEN_RE.is_match(&result) {
                return strip_thinking(&result);
            }
            return result.trim().to_string();
        }
        // 2. No closing tag: find first tool candidate
        let lower = after_open.to_lowercase();
        let mut min_pos: Option<usize> = None;
        for &cand in TOOL_START_CANDIDATES {
            if let Some(pos) = lower.find(cand) {
                if min_pos.is_none_or(|p| pos < p) {
                    min_pos = Some(pos);
                }
            }
        }
        if let Some(pos) = min_pos {
            let before = &text[..open_m.start()];
            let after = &after_open[pos..];
            return format!("{}{}", before, after).trim().to_string();
        }
        let before = text[..open_m.start()].trim();
        if !before.is_empty() {
            return before.to_string();
        }
    }
    text.to_string()
}

/// Cleans an extracted XML/DSML parameter value.
pub fn clean_param_val(val: &str, is_code_param: bool) -> String {
    let (s, has_cdata) = if val.contains("<![CDATA[") {
        if let Some(caps) = CDATA_RE.captures(val) {
            (caps.get(1).map_or("", |m| m.as_str()), true)
        } else {
            (val, false)
        }
    } else {
        (val, false)
    };

    let processed = if !has_cdata && s.contains('&') {
        html_escape::decode_html_entities(s).to_string()
    } else {
        s.to_string()
    };

    if is_code_param {
        let mut res = processed.as_str();
        if res.starts_with("\r\n") {
            res = &res[2..];
        } else if res.starts_with('\n') {
            res = &res[1..];
        }
        if res.ends_with("\r\n") {
            res = &res[..res.len() - 2];
        } else if res.ends_with('\n') {
            res = &res[..res.len() - 1];
        }
        res.to_string()
    } else {
        processed.trim().to_string()
    }
}

/// Applies heuristic repairs to malformed JSON strings.
pub fn clean_json_str(s: &str) -> String {
    let norm = normalize_truncated_json_prefix(s);
    let s = norm.as_ref();
    let mut cleaned = if s.contains('\'') {
        if !s.contains('"') {
            s.replace('\'', "\"")
        } else {
            let step1 = SINGLE_QUOTE_KEYS_RE.replace_all(s, r#"$1"$2"$3"#);
            SINGLE_QUOTE_VALS_RE
                .replace_all(&step1, |caps: &regex::Captures| {
                    let prefix = &caps[1];
                    let val = &caps[2];
                    let suffix = &caps[3];
                    let val_escaped = val.replace("\\\"", "\"").replace('"', "\\\"");
                    format!("{}\"{}\"{}", prefix, val_escaped, suffix)
                })
                .to_string()
        }
    } else {
        s.to_string()
    };

    if cleaned.contains(',') {
        cleaned = TRAILING_COMMA_RE.replace_all(&cleaned, "$1").to_string();
    }
    if cleaned.contains("True") {
        cleaned = PY_TRUE_RE.replace_all(&cleaned, "true").to_string();
    }
    if cleaned.contains("False") {
        cleaned = PY_FALSE_RE.replace_all(&cleaned, "false").to_string();
    }
    if cleaned.contains("None") {
        cleaned = PY_NONE_RE.replace_all(&cleaned, "null").to_string();
    }
    if serde_json::from_str::<serde_json::Value>(&cleaned).is_err() && cleaned.contains('"') {
        cleaned = repair_unescaped_quotes(&cleaned);
    }
    cleaned
}

fn is_valid_closing_quote(
    chars: &[(usize, char)],
    curr_idx: usize,
    is_key: bool,
    in_object: bool,
) -> bool {
    let mut j = curr_idx + 1;
    while j < chars.len() && chars[j].1.is_whitespace() {
        j += 1;
    }
    if j >= chars.len() {
        return true;
    }
    let next_ch = chars[j].1;

    if is_key {
        return next_ch == ':';
    }

    if next_ch == ':' {
        return false;
    }

    if next_ch == '}' || next_ch == ']' {
        return true;
    }

    if next_ch == ',' {
        if !in_object {
            return true;
        }
        let mut k = j + 1;
        while k < chars.len() && chars[k].1.is_whitespace() {
            k += 1;
        }
        if k >= chars.len() {
            return true;
        }
        if chars[k].1 == '}' {
            return true;
        }
        if chars[k].1 == '"' || chars[k].1 == '\'' {
            let quote = chars[k].1;
            k += 1;
            let mut escape = false;
            while k < chars.len() {
                let c = chars[k].1;
                if c == '\\' && !escape {
                    escape = true;
                } else if c == quote && !escape {
                    k += 1;
                    break;
                } else {
                    escape = false;
                }
                k += 1;
            }
            while k < chars.len() && chars[k].1.is_whitespace() {
                k += 1;
            }
            if k < chars.len() && chars[k].1 == ':' {
                return true;
            }
        }
        return false;
    }

    false
}

/// Heuristic repair for unescaped double quotes inside JSON string literals.
///
/// Models frequently emit code snippets or shell commands with raw unescaped quotes:
/// `{"name": "str_replace", "arguments": {"new_str": "print("Hello world")"}}`
/// This function identifies unescaped internal quotes and escapes them (`\"`),
/// while preserving legitimate structural JSON quotes and already-escaped sequences.
pub fn repair_unescaped_quotes(s: &str) -> String {
    if serde_json::from_str::<serde_json::Value>(s).is_ok() {
        return s.to_string();
    }

    let chars: Vec<(usize, char)> = s.char_indices().collect();
    if chars.is_empty() {
        return s.to_string();
    }

    let mut out = String::with_capacity(s.len() + 16);
    let mut in_string = false;
    let mut is_key = false;
    let mut stack: Vec<char> = Vec::new();
    let mut expect_key = false;
    let mut i = 0;

    while i < chars.len() {
        let (_, c) = chars[i];

        if !in_string {
            match c {
                '{' => {
                    stack.push('{');
                    expect_key = true;
                    out.push(c);
                }
                '[' => {
                    stack.push('[');
                    expect_key = false;
                    out.push(c);
                }
                '}' => {
                    if let Some(top) = stack.last() {
                        if *top == '{' {
                            stack.pop();
                        }
                    }
                    expect_key = false;
                    out.push(c);
                }
                ']' => {
                    if let Some(top) = stack.last() {
                        if *top == '[' {
                            stack.pop();
                        }
                    }
                    expect_key = false;
                    out.push(c);
                }
                ':' => {
                    expect_key = false;
                    out.push(c);
                }
                ',' => {
                    if stack.last() == Some(&'{') {
                        expect_key = true;
                    }
                    out.push(c);
                }
                '"' => {
                    in_string = true;
                    is_key = stack.last() == Some(&'{') && expect_key;
                    out.push(c);
                }
                _ => {
                    out.push(c);
                }
            }
            i += 1;
        } else {
            if c == '\\' {
                out.push('\\');
                if i + 1 < chars.len() {
                    i += 1;
                    out.push(chars[i].1);
                }
                i += 1;
                continue;
            }

            if c == '"' {
                let is_closing =
                    is_valid_closing_quote(&chars, i, is_key, stack.last() == Some(&'{'));
                if is_closing {
                    in_string = false;
                    if is_key {
                        expect_key = false;
                    }
                    out.push('"');
                } else {
                    out.push_str(r#"\""#);
                }
                i += 1;
            } else {
                out.push(c);
                i += 1;
            }
        }
    }

    out
}

/// Escapes raw control characters (< 0x20, like \n, \t) inside unescaped JSON string literals.
pub fn escape_json_control_chars(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut in_string = false;
    let mut escape = false;

    for c in s.chars() {
        if c == '"' && !escape {
            in_string = !in_string;
            out.push(c);
        } else if c == '\\' && in_string {
            escape = !escape;
            out.push(c);
        } else {
            if in_string && !escape {
                match c {
                    '\n' => out.push_str(r#"\n"#),
                    '\r' => out.push_str(r#"\r"#),
                    '\t' => out.push_str(r#"\t"#),
                    c if (c as u32) < 0x20 => {
                        use std::fmt::Write;
                        let _ = write!(out, r#"\u{:04x}"#, c as u32);
                    }
                    _ => out.push(c),
                }
            } else {
                out.push(c);
            }
            escape = false;
        }
    }
    out
}

/// Safely parses JSON with fallback to heuristic repair, returning None on failure.
pub fn safe_json_loads(s: &str) -> Option<serde_json::Value> {
    if s.trim().is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
        return Some(v);
    }
    let escaped = escape_json_control_chars(s);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&escaped) {
        return Some(v);
    }
    let repaired_quotes = repair_unescaped_quotes(&escaped);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired_quotes) {
        return Some(v);
    }
    let repaired = clean_json_str(&escaped);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired) {
        return Some(v);
    }
    let repaired_orig = clean_json_str(s);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired_orig) {
        return Some(v);
    }
    let repaired_orig_quotes = repair_unescaped_quotes(&repaired_orig);
    serde_json::from_str::<serde_json::Value>(&repaired_orig_quotes).ok()
}

/// Extracts all balanced JSON object candidates from arbitrary text using bracket-depth tracking.
pub fn extract_json_objects(text: &str) -> Vec<String> {
    let norm = normalize_truncated_json_prefix(text);
    let text = norm.as_ref();
    if !text.contains('{') {
        return Vec::new();
    }
    let mut results = Vec::new();
    let mut stack: Vec<char> = Vec::new();
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut quote_char: Option<char> = None;
    let mut escape = false;

    for (i, char) in text.char_indices() {
        if (char == '"' || char == '\'') && !escape {
            if !in_string {
                in_string = true;
                quote_char = Some(char);
            } else if quote_char == Some(char) {
                in_string = false;
                quote_char = None;
            }
        } else if char == '\\' && in_string {
            escape = !escape;
            continue;
        } else if !in_string {
            if char == '{' {
                if stack.is_empty() {
                    start = Some(i);
                }
                stack.push('{');
            } else if char == '[' {
                if !stack.is_empty() {
                    stack.push('[');
                }
            } else if char == '}' {
                if let Some(&top) = stack.last() {
                    if top == '{' {
                        stack.pop();
                        if stack.is_empty() {
                            if let Some(s) = start {
                                let end = i + char.len_utf8();
                                results.push(text[s..end].to_string());
                                start = None;
                            }
                        }
                    } else if stack.contains(&'{') {
                        while let Some(b) = stack.pop() {
                            if b == '{' {
                                break;
                            }
                        }
                        if stack.is_empty() {
                            if let Some(s) = start {
                                let end = i + char.len_utf8();
                                results.push(text[s..end].to_string());
                                start = None;
                            }
                        }
                    }
                }
            } else if char == ']' {
                if let Some(&top) = stack.last() {
                    if top == '[' {
                        stack.pop();
                    }
                }
            }
        }
        if escape {
            escape = false;
        }
    }

    // Auto-repair truncated tail if JSON generation was cut off mid-stream
    if !stack.is_empty() {
        if let Some(s) = start {
            let mut tail = text[s..].to_string();
            if in_string {
                if let Some(qc) = quote_char {
                    if escape || tail.ends_with('\\') {
                        while tail.ends_with('\\') {
                            tail.pop();
                        }
                    }
                    tail.push(qc);
                }
            }
            let trimmed = tail.trim_end();
            if trimmed.ends_with(':') {
                tail = format!("{}\"\"", trimmed);
            } else if let Some(stripped) = trimmed.strip_suffix(',') {
                tail = stripped.to_string();
            }
            for b in stack.iter().rev() {
                if *b == '[' {
                    tail.push(']');
                } else if *b == '{' {
                    tail.push('}');
                }
            }
            results.push(tail);
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repair_unescaped_quotes_code_payload() {
        let input = r#"{"name": "str_replace", "arguments": {"new_str": "print("Hello world")"}}"#;
        let repaired = repair_unescaped_quotes(input);
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["arguments"]["new_str"], "print(\"Hello world\")");
    }

    #[test]
    fn test_repair_unescaped_quotes_multi_field() {
        let input = r#"{"name": "str_replace", "arguments": {"new_str": "print("Hello")", "path": "main.py"}}"#;
        let val = safe_json_loads(input).unwrap();
        assert_eq!(val["arguments"]["new_str"], "print(\"Hello\")");
        assert_eq!(val["arguments"]["path"], "main.py");
    }

    #[test]
    fn test_repair_unescaped_quotes_shell_command() {
        let input = r#"{"command": "echo "hello world" >> log.txt"}"#;
        let val = safe_json_loads(input).unwrap();
        assert_eq!(val["command"], "echo \"hello world\" >> log.txt");
    }

    #[test]
    fn test_repair_preserves_already_escaped() {
        let input = r#"{"new_str": "print(\"Hello world\")"}"#;
        let repaired = repair_unescaped_quotes(input);
        assert_eq!(repaired, input);
    }
}
