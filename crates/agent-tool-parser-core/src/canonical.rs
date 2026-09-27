// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

use serde_json::Value;
use std::collections::BTreeMap;

/// Recursively sorts dictionary keys in a serde_json::Value.
/// Ensures that identical payloads always yield identical JSON byte representations,
/// maximizing KV-cache prefix hits across inference engines (vLLM, SGLang, Ollama, DeepSeek).
pub fn canonical_sort_keys(val: &Value) -> Value {
    match val {
        Value::Object(map) => {
            let mut sorted: BTreeMap<String, Value> = BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), canonical_sort_keys(v));
            }
            Value::Object(serde_json::Map::from_iter(sorted))
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonical_sort_keys).collect()),
        _ => val.clone(),
    }
}

/// Serializes value to deterministic canonical JSON with sorted keys.
pub fn canonical_json_dumps(val: &Value) -> String {
    let sorted = canonical_sort_keys(val);
    serde_json::to_string(&sorted).unwrap_or_default()
}

/// Parses raw JSON string, sorts keys recursively, and outputs compact canonical JSON.
pub fn canonicalize_arguments_string(args_str: &str) -> String {
    let trimmed = args_str.trim();
    if !((trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']')))
    {
        return trimmed.to_string();
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(parsed) => canonical_json_dumps(&parsed),
        Err(_) => trimmed.to_string(),
    }
}

/// Stably sorts and canonicalizes a slice of OpenAI tool call JSON objects.
pub fn canonicalize_tool_calls(calls: &[Value]) -> Vec<Value> {
    let mut aligned: Vec<Value> = Vec::with_capacity(calls.len());
    for item in calls {
        if let Value::Object(mut map) = item.clone() {
            if let Some(Value::Object(mut func_map)) = map.remove("function") {
                if let Some(args_val) = func_map.get("arguments") {
                    let canon_args = match args_val {
                        Value::String(s) => canonicalize_arguments_string(s),
                        Value::Object(_) | Value::Array(_) => canonical_json_dumps(args_val),
                        _ => args_val.to_string(),
                    };
                    func_map.insert("arguments".to_string(), Value::String(canon_args));
                }
                map.insert("function".to_string(), Value::Object(func_map));
            }
            aligned.push(Value::Object(map));
        } else {
            aligned.push(item.clone());
        }
    }

    // Stable sort by function name, then by id
    aligned.sort_by(|a, b| {
        let name_a = a
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("");
        let name_b = b
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("");
        let id_a = a.get("id").and_then(|i| i.as_str()).unwrap_or("");
        let id_b = b.get("id").and_then(|i| i.as_str()).unwrap_or("");
        (name_a, id_a).cmp(&(name_b, id_b))
    });

    aligned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_sort_keys_nested() {
        let raw = r#"{"z": 10, "a": {"b": 2, "a": 1}}"#;
        let v: Value = serde_json::from_str(raw).unwrap();
        let canon = canonical_json_dumps(&v);
        assert_eq!(canon, r#"{"a":{"a":1,"b":2},"z":10}"#);
    }

    #[test]
    fn test_canonicalize_arguments_string() {
        let raw = r#"{ "b": 2, "a": 1 }"#;
        let canon = canonicalize_arguments_string(raw);
        assert_eq!(canon, r#"{"a":1,"b":2}"#);

        assert_eq!(canonicalize_arguments_string("not_json"), "not_json");
    }

    #[test]
    fn test_canonicalize_tool_calls() {
        let calls_json = serde_json::json!([
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": "{\"path\":\"b.py\",\"content\":\"bar\"}"
                }
            },
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "arguments": "{\"z\":1,\"a\":2}"
                }
            }
        ]);
        let calls = calls_json.as_array().unwrap();
        let canon = canonicalize_tool_calls(calls);
        assert_eq!(canon.len(), 2);
        // edit_file comes before write_file alphabetically
        assert_eq!(canon[0]["function"]["name"].as_str().unwrap(), "edit_file");
        assert_eq!(
            canon[0]["function"]["arguments"].as_str().unwrap(),
            r#"{"a":2,"z":1}"#
        );
        assert_eq!(canon[1]["function"]["name"].as_str().unwrap(), "write_file");
    }
}
