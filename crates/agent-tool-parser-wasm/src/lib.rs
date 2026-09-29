// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

use agent_tool_parser_core::{
    cleaners, parse_tool_call as core_parse_call, parse_tool_calls as core_parse_calls,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn parse_tool_call(text: &str) -> Result<JsValue, JsValue> {
    match core_parse_call(text) {
        Ok(call) => {
            serde_wasm_bindgen::to_value(&call).map_err(|e| JsValue::from_str(&e.to_string()))
        }
        Err(e) => Err(JsValue::from_str(&e.message)),
    }
}

#[wasm_bindgen]
pub fn parse_tool_calls(text: &str) -> Result<JsValue, JsValue> {
    match core_parse_calls(text) {
        Ok(calls) => {
            serde_wasm_bindgen::to_value(&calls).map_err(|e| JsValue::from_str(&e.to_string()))
        }
        Err(e) => Err(JsValue::from_str(&e.message)),
    }
}

#[wasm_bindgen]
pub fn extract_raw_tool_calls(text: &str) -> Result<JsValue, JsValue> {
    let calls = agent_tool_parser_core::extract_raw_tool_calls(text);
    serde_wasm_bindgen::to_value(&calls).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn try_extract_raw_tool_call(text: &str) -> Result<JsValue, JsValue> {
    let call = agent_tool_parser_core::try_extract_raw_tool_call(text);
    serde_wasm_bindgen::to_value(&call).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn clean_json_str(s: &str) -> String {
    cleaners::clean_json_str(s)
}

#[wasm_bindgen]
pub fn strip_thinking(text: &str) -> String {
    cleaners::strip_thinking(text)
}

#[wasm_bindgen]
pub fn canonicalize_json(text: &str) -> String {
    agent_tool_parser_core::canonicalize_arguments_string(text)
}

#[wasm_bindgen]
pub fn repair_unescaped_quotes(text: &str) -> String {
    cleaners::repair_unescaped_quotes(text)
}

#[wasm_bindgen]
pub fn detect_vector_engine() -> String {
    agent_tool_parser_core::detect_vector_engine()
        .as_str()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_clean_json_str() {
        let raw = "{'action': 'bash', 'flag': True,}";
        assert_eq!(
            clean_json_str(raw),
            "{\"action\": \"bash\", \"flag\": true}"
        );
    }

    #[test]
    fn test_wasm_strip_thinking() {
        let text = "<think>Analyzing query...</think>Action: run";
        assert_eq!(strip_thinking(text), "Action: run");
    }

    #[test]
    fn test_wasm_canonicalize_json() {
        let raw = r#"{"z": 10, "a": 2}"#;
        assert_eq!(canonicalize_json(raw), r#"{"a":2,"z":10}"#);
    }

    #[test]
    fn test_wasm_repair_unescaped_quotes() {
        let raw = r#"{"cmd": "echo "hello""}"#;
        assert_eq!(repair_unescaped_quotes(raw), r#"{"cmd": "echo \"hello\""}"#);
    }

    #[test]
    fn test_wasm_detect_vector_engine() {
        let engine = detect_vector_engine();
        assert!(!engine.is_empty());
    }
}
