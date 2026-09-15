// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT
//
// Degenerate, Pathological, and Adversarial Input Test Suite for agent-tool-parser-core
//
// Provenance & IP Hygiene Notice:
// All test cases in this file are original clean-room synthetic cases targeting
// degenerate input texts, malformed streams, control characters, ReDoS resistance,
// and adversarial prompt structures. Authored under the MIT License.

use agent_tool_parser_core::{
    parse_tool_call, parse_tool_calls, try_parse_tool_call, try_parse_tool_calls,
};
use serde_json::json;
use std::time::Instant;

// =========================================================================
// 1. Empty, Whitespace, Control Characters & Non-Printable Bytes
// =========================================================================

#[test]
fn test_degenerate_empty_and_whitespace() {
    let cases = ["", "   ", "\t", "\n\n\n", "\r\n\r\n", "   \t  \r\n   "];
    for inp in cases {
        assert!(try_parse_tool_calls(inp).is_empty());
        assert!(try_parse_tool_call(inp).is_none());
        assert!(parse_tool_call(inp).is_err());
    }
}

#[test]
fn test_degenerate_invisible_unicode_and_bom() {
    let text =
        "\u{feff}\u{200b}\u{200c}{\"tool\": \"bash\", \"command\": \"ls -la\"}\u{200d}\u{200e}";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert_eq!(call.args["command"], "ls -la");
}

#[test]
fn test_degenerate_ansi_escape_code_wrappers() {
    let text = "\x1b[31;1m<invoke name=\"bash\"><parameter name=\"cmd\">cargo test</parameter></invoke>\x1b[0m";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert_eq!(call.args["command"], "cargo test");
}

#[test]
fn test_degenerate_embedded_null_byte() {
    let text = "{\"tool\": \"bash\", \"command\": \"echo hello\0world\"}";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert!(call.args["command"]
        .as_str()
        .unwrap()
        .contains("hello\0world"));

    let text_escaped = "{\"tool\": \"bash\", \"command\": \"echo hello\\u0000world\"}";
    let call_escaped = parse_tool_call(text_escaped).unwrap();
    assert_eq!(call_escaped.name, "bash");
    assert!(call_escaped.args["command"]
        .as_str()
        .unwrap()
        .contains("hello\0world"));
}

// =========================================================================
// 2. Streaming Truncation & Token Cut-Offs (EOF at Arbitrary Positions)
// =========================================================================

#[test]
fn test_degenerate_eof_after_tag_opening() {
    let cases = [
        "<invoke",
        "<invoke ",
        "<tool_call",
        "<｜DSML｜",
        "<｜DSML｜tool",
        "<｜tool calls begin｜>",
        "<｜tool calls begin｜><｜tool call begin｜>",
    ];
    for inp in cases {
        assert!(try_parse_tool_calls(inp).is_empty());
        assert!(parse_tool_call(inp).is_err());
    }
}

#[test]
fn test_degenerate_eof_after_json_colon() {
    let text = "{\"tool\": \"bash\", \"command\":";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert_eq!(call.args["command"], "");
}

#[test]
fn test_degenerate_eof_inside_json_array() {
    let text = "{\"tool\": \"process\", \"items\": [\"alpha\", \"beta\", ";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "process");
    assert_eq!(call.args["items"], json!(["alpha", "beta"]));
}

#[test]
fn test_degenerate_eof_with_trailing_escape_backslash() {
    let text = "{\"tool\": \"bash\", \"command\": \"echo hello\\";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert_eq!(call.args["command"], "echo hello");
}

#[test]
fn test_degenerate_eof_inside_cdata() {
    let code = "def process():\n    for i in range(10):\n        print(i)";
    let text = format!(
        "<invoke name=\"write_file\"><parameter name=\"content\"><![CDATA[{}",
        code
    );
    let call = parse_tool_call(&text).unwrap();
    assert_eq!(call.name, "write_file");
    assert_eq!(call.args["content"].as_str().unwrap().trim(), code.trim());
}

// =========================================================================
// 3. Malformed XML & Tag Pathologies
// =========================================================================

#[test]
fn test_degenerate_unclosed_sequential_parameters() {
    let text = r#"<invoke name="deploy">
<parameter name="env">staging
<parameter name="version">v2.1.0
<parameter name="dry_run">true
</invoke>"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "deploy");
    assert_eq!(call.args["env"], "staging");
    assert_eq!(call.args["version"], "v2.1.0");
    assert_eq!(call.args["dry_run"], "true");
}

#[test]
fn test_degenerate_empty_parameter_tags() {
    let text = "<invoke name=\"bash\"><parameter name=\"cmd\"></parameter></invoke>";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert_eq!(call.args["command"], "");
}

#[test]
fn test_degenerate_duplicate_parameter_names() {
    let text = r#"<invoke name="bash">
<parameter name="cmd">echo first</parameter>
<parameter name="cmd">echo second</parameter>
</invoke>"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert_eq!(call.args["command"], "echo second");
}

#[test]
fn test_degenerate_mismatched_closing_tag() {
    let text = "<invoke name=\"bash\"><parameter name=\"cmd\">pytest</parameter></tool>";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert_eq!(call.args["command"], "pytest");
}

#[test]
fn test_degenerate_dsml_parameter_typo_closing_tag() {
    let text = r#"<｜DSML｜tool_calls>
<｜DSML｜tool name=search>
<｜DSML｜parameter name=pattern>vector databases</｜DSML｜tool_name>
<｜DSML｜parameter name=limit>10</｜DSML｜tool_name>
</｜DSML｜tool>
</｜DSML｜tool_calls>"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "search");
    assert_eq!(call.args["pattern"], "vector databases");
    assert_eq!(call.args["limit"], 10);

    let text_invoke = "<invoke name=\"bash\"><parameter name=\"cmd\">pytest</tool_name></invoke>";
    let call_invoke = parse_tool_call(text_invoke).unwrap();
    assert_eq!(call_invoke.name, "bash");
    assert_eq!(call_invoke.args["command"], "pytest");
}

#[test]
fn test_degenerate_html_entities_decoding() {
    let text = "<invoke name=\"bash\"><parameter name=\"cmd\">cat &lt; input.txt &amp;&amp; echo &quot;hello&quot; &#39;world&#39;</parameter></invoke>";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert_eq!(
        call.args["command"],
        "cat < input.txt && echo \"hello\" 'world'"
    );
}

// =========================================================================
// 4. Corrupted & Invalid JSON Handling
// =========================================================================

#[test]
fn test_degenerate_single_quotes_with_inner_double_quotes() {
    let text = "{'tool': 'bash', 'command': 'echo \"Hello World\"'}";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert_eq!(call.args["command"], "echo \"Hello World\"");
}

#[test]
fn test_headless_json_minimax_prefix_omissions() {
    // 1. Omitted `{"` or `{` with key name `tool"`:
    let text1 = r#"tool": "bash", "command": "ls -la"}"#;
    let call1 = parse_tool_call(text1).unwrap();
    assert_eq!(call1.name, "bash");
    assert_eq!(call1.args["command"], "ls -la");

    // 2. Omitted `{"tool`:
    let text2 = r#": "read_file", "path": "src/main.rs"}"#;
    let call2 = parse_tool_call(text2).unwrap();
    assert_eq!(call2.name, "read_file");
    assert_eq!(call2.args["path"], "src/main.rs");

    let text3 = r#"": "read_file", "path": "src/main.rs"}"#;
    let call3 = parse_tool_call(text3).unwrap();
    assert_eq!(call3.name, "read_file");
    assert_eq!(call3.args["path"], "src/main.rs");

    // 3. Omitted `{` with quoted key:
    let text4 = r#""tool": "bash", "command": "ls -la"}"#;
    let call4 = parse_tool_call(text4).unwrap();
    assert_eq!(call4.name, "bash");
    assert_eq!(call4.args["command"], "ls -la");

    // 4. Omitted `{` with nested arguments:
    let text5 = r#""name": "read_file", "arguments": {"path": "src/main.rs"}}"#;
    let call5 = parse_tool_call(text5).unwrap();
    assert_eq!(call5.name, "read_file");
    assert_eq!(call5.args["path"], "src/main.rs");

    let text6 = r#"name": "read_file", "arguments": {"path": "src/main.rs"}}"#;
    let call6 = parse_tool_call(text6).unwrap();
    assert_eq!(call6.name, "read_file");
    assert_eq!(call6.args["path"], "src/main.rs");

    // 5. In markdown codeblock:
    let text7 = "```json\ntool\": \"bash\", \"command\": \"ls -la\"}\n```";
    let call7 = parse_tool_call(text7).unwrap();
    assert_eq!(call7.name, "bash");
    assert_eq!(call7.args["command"], "ls -la");

    // 6. With leading conversational text:
    let text8 = "Here is the tool call: : \"read_file\", \"path\": \"src/main.rs\"}";
    let call8 = parse_tool_call(text8).unwrap();
    assert_eq!(call8.name, "read_file");
    assert_eq!(call8.args["path"], "src/main.rs");
}

#[test]
fn test_degenerate_empty_or_whitespace_tool_name_rejected() {
    let cases = [
        "{\"tool\": \"\", \"command\": \"ls\"}",
        "{\"tool\": \"   \", \"command\": \"ls\"}",
        "{\"name\": \"\", \"arguments\": {}}",
        "<invoke name=\"\"><parameter name=\"cmd\">ls</parameter></invoke>",
        "<invoke name=\"   \"><parameter name=\"cmd\">ls</parameter></invoke>",
    ];
    for inp in cases {
        assert!(try_parse_tool_calls(inp).is_empty());
        assert!(parse_tool_call(inp).is_err());
    }
}

#[test]
fn test_degenerate_non_string_or_null_tool_name_rejected() {
    let cases = [
        "{\"tool\": null, \"command\": \"ls\"}",
        "{\"tool\": 12345, \"command\": \"ls\"}",
        "{\"tool\": true, \"command\": \"ls\"}",
        "{\"tool\": [\"bash\"], \"command\": \"ls\"}",
        "{\"name\": null, \"arguments\": {}}",
        "{\"name\": 42, \"arguments\": {}}",
    ];
    for inp in cases {
        assert!(try_parse_tool_calls(inp).is_empty());
        assert!(parse_tool_call(inp).is_err());
    }
}

#[test]
fn test_degenerate_empty_objects_and_arrays() {
    let cases = ["{}", "[]", "   {}   ", "   []   "];
    for inp in cases {
        assert!(try_parse_tool_calls(inp).is_empty());
        assert!(parse_tool_call(inp).is_err());
    }
}

#[test]
fn test_degenerate_unescaped_literal_newlines() {
    let text = "{\"tool\": \"bash\", \"command\": \"echo line 1\necho line 2\necho line 3\"}";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert_eq!(
        call.args["command"],
        "echo line 1\necho line 2\necho line 3"
    );
}

// =========================================================================
// 5. Code, Math & Syntax Collisions (False Positive Immunity)
// =========================================================================

#[test]
fn test_degenerate_math_inequalities_collision() {
    let text = r#"Here is the validation logic in Python:
def check(x, y):
    if x < 10 and y > 20:
        return x < 5 or y > 50
    return False"#;
    assert!(try_parse_tool_calls(text).is_empty());
    assert!(parse_tool_call(text).is_err());
}

#[test]
fn test_degenerate_cpp_templates_collision() {
    let text = "Use this type definition: std::vector<std::pair<int, int>> lookup_table;";
    assert!(try_parse_tool_calls(text).is_empty());
    assert!(parse_tool_call(text).is_err());
}

#[test]
fn test_degenerate_react_keyword_collision() {
    let text = "In conclusion, Action: we should consider refactoring the parser next sprint.";
    assert!(try_parse_tool_calls(text).is_empty());
    assert!(parse_tool_call(text).is_err());
}

// =========================================================================
// 6. Thinking Blocks Pathologies
// =========================================================================

#[test]
fn test_degenerate_thinking_block_retracted_proposal() {
    let text = r#"<think>
I should probably run:
<tool_call>
{"name": "bash", "arguments": {"cmd": "rm -rf /"}}
</tool_call>
Wait, that is destructive and dangerous. I will not run that.
</think>
I cannot perform destructive actions."#;
    assert!(try_parse_tool_calls(text).is_empty());
    assert!(parse_tool_call(text).is_err());
}

#[test]
fn test_degenerate_multiple_thinking_blocks_with_real_call() {
    let text = r#"<think>first contemplation</think>
Intermediate explanation.
<think>second contemplation</think>
<invoke name="search">
<parameter name="query">rust parser</parameter>
</invoke>"#;
    let calls = parse_tool_calls(text).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "search");
    assert_eq!(calls[0].args["pattern"], "rust parser");
}

// =========================================================================
// 7. Stress & ReDoS Resilience
// =========================================================================

#[test]
fn test_degenerate_massive_string_without_catastrophic_backtracking() {
    let payload = "x".repeat(50_000);
    let text = format!(
        "<invoke name=\"bash\"><parameter name=\"cmd\">{}</parameter></invoke>",
        payload
    );
    let start = Instant::now();
    let call = parse_tool_call(&text).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(call.name, "bash");
    assert_eq!(call.args["command"].as_str().unwrap().len(), 50_000);
    assert!(
        elapsed.as_millis() < 500,
        "ReDoS vulnerability detected: {:?}",
        elapsed
    );
}
