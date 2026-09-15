//! `agent-tool-parser-core`: Fast, zero-dependency tool call parser for LLM agent outputs.
//!
//! Copyright (c) 2026 InBoost Team.
//! SPDX-License-Identifier: MIT
//! Licensed under the MIT License. Provided "AS IS" without warranty of any kind.
//! See LICENSE and README.md for full terms and execution safety disclaimers.

pub mod cleaners;
pub mod models;
pub mod parser;

pub use cleaners::{
    clean_json_str, clean_param_val, extract_json_objects, normalize_truncated_json_prefix,
    safe_json_loads, strip_thinking,
};
pub use models::{ToolCall, ToolError, ToolParserConfig};
pub use parser::{
    parse_tool_call, parse_tool_calls, try_parse_tool_call, try_parse_tool_calls, ToolParser,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsml_unquoted_and_truncated() {
        let text = r#"<｜DSML｜tool_calls>
<｜DSML｜tool name=search>
<｜DSML｜parameter name=pattern string=true>ignore-paths</｜DSML｜parameter>
<｜DSML｜parameter name=path string=true>.</｜DSML｜parameter>
</｜DSML｜invoca"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "search");
        assert_eq!(call.args["pattern"], "ignore-paths");
        assert_eq!(call.args["path"], ".");
    }

    #[test]
    fn test_deepseek_native_tokens() {
        let text = r#"<｜tool calls begin｜><｜tool call begin｜>function=search<｜tool sep｜>{"pattern": "ignore-paths", "path": "."}<｜tool call end｜><｜tool calls end｜>"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "search");
        assert_eq!(call.args["pattern"], "ignore-paths");
        assert_eq!(call.args["path"], ".");
    }

    #[test]
    fn test_cdata_code_preservation() {
        let code = "    def test():\n        x = 1 < 2 && 3 > 0\n        return x";
        let text = format!(
            "<invoke name=\"write_file\">\n<parameter name=\"path\">test.py</parameter>\n<parameter name=\"content\"><![CDATA[\n{}\n]]></parameter>\n</invoke>",
            code
        );
        let call = parse_tool_call(&text).unwrap();
        assert_eq!(call.name, "write_file");
        assert_eq!(call.args["content"], code);
    }

    #[test]
    fn test_hermes_xml_tool_call() {
        let text = r#"<tool_call>
{"name": "search", "arguments": {"query": "parse_tool", "path": "src"}}
</tool_call>"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "search");
        assert_eq!(call.args["pattern"], "parse_tool");
        assert_eq!(call.args["path"], "src");
    }

    #[test]
    fn test_qwen_agent() {
        let text = r#"<action_name>bash<action_args>{"cmd": "cargo test"}<action_end>"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "bash");
        assert_eq!(call.args["command"], "cargo test");
    }

    #[test]
    fn test_chatglm_agent() {
        let text = r#"FUNCTION: bash ARGS: {"command": "ls -la"}"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "bash");
        assert_eq!(call.args["command"], "ls -la");
    }

    #[test]
    fn test_mistral_tool_call() {
        let text = r#"[TOOL_CALLS] [{"name": "read_file", "arguments": {"path": "README.md"}}]"#;
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].args["path"], "README.md");
    }

    #[test]
    fn test_llama_python_tag() {
        let text = r#"<python_tag>search(pattern="foo", path="bar")</python_tag>"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "search");
        assert_eq!(call.args["pattern"], "foo");
        assert_eq!(call.args["path"], "bar");
    }

    #[test]
    fn test_react_format() {
        let text = "Thought: I should run tests.\nAction: bash\nAction Input: cargo test";
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "bash");
        assert_eq!(call.args["command"], "cargo test");
    }

    #[test]
    fn test_json_repair_and_strip_thinking() {
        let text = "<think>Let me find the file</think>\n```json\n{'path': 'src/lib.rs', 'offset': 10}\n```";
        let cleaned = strip_thinking(text);
        assert!(!cleaned.contains("<think>"));
        let repaired = cleaners::clean_json_str("{'a': 1, 'b': 'c'}");
        assert_eq!(repaired, r#"{"a": 1, "b": "c"}"#);
    }
}
