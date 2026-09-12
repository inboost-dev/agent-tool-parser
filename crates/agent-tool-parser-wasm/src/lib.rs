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
pub fn clean_json_str(s: &str) -> String {
    cleaners::clean_json_str(s)
}

#[wasm_bindgen]
pub fn strip_thinking(text: &str) -> String {
    cleaners::strip_thinking(text)
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
}
