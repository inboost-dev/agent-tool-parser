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
    clean_json_str, parse_tool_call, parse_tool_calls, try_parse_tool_call, try_parse_tool_calls,
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
    // Pattern 1: Omitted opening curly brace before key (tool/name/action)
    let p1_1 = r#"tool": "bash", "command": "git diff"}"#;
    let call1_1 = parse_tool_call(p1_1).unwrap();
    assert_eq!(call1_1.name, "bash");
    assert_eq!(call1_1.args["command"], "git diff");

    let p1_2 = r#"name": "bash", "input": {"command": "ls"}}"#;
    let call1_2 = parse_tool_call(p1_2).unwrap();
    assert_eq!(call1_2.name, "bash");
    assert_eq!(call1_2.args["command"], "ls");

    let p1_3 = r#"action": "read_file", "path": "main.py"}"#;
    let call1_3 = parse_tool_call(p1_3).unwrap();
    assert_eq!(call1_3.name, "read_file");
    assert_eq!(call1_3.args["path"], "main.py");

    // Pattern 2: Omitted brace and opening quote
    let p2_1 = r#""tool": "read_file", "path": "src/app.py", "offset": 1, "limit": 100}"#;
    let call2_1 = parse_tool_call(p2_1).unwrap();
    assert_eq!(call2_1.name, "read_file");
    assert_eq!(call2_1.args["path"], "src/app.py");
    assert_eq!(call2_1.args["offset"], 1);
    assert_eq!(call2_1.args["limit"], 100);

    let p2_2 = r#""name": "bash", "command": "pytest"}"#;
    let call2_2 = parse_tool_call(p2_2).unwrap();
    assert_eq!(call2_2.name, "bash");
    assert_eq!(call2_2.args["command"], "pytest");

    // Pattern 3: Omitted brace and key name (direct colon with value)
    let p3_1 =
        r#"": "read_file", "path": "sklearn/impute/_iterative.py", "offset": 1, "limit": 100}"#;
    let call3_1 = parse_tool_call(p3_1).unwrap();
    assert_eq!(call3_1.name, "read_file");
    assert_eq!(call3_1.args["path"], "sklearn/impute/_iterative.py");
    assert_eq!(call3_1.args["offset"], 1);
    assert_eq!(call3_1.args["limit"], 100);

    let p3_2 = r#"": "search", "pattern": "initial_strategy", "path": "."}"#;
    let call3_2 = parse_tool_call(p3_2).unwrap();
    assert_eq!(call3_2.name, "search");
    assert_eq!(call3_2.args["pattern"], "initial_strategy");
    assert_eq!(call3_2.args["path"], ".");

    let p3_3 =
        r#": "read_file", "path": "sklearn/impute/_iterative.py", "offset": 1, "limit": 100}"#;
    let call3_3 = parse_tool_call(p3_3).unwrap();
    assert_eq!(call3_3.name, "read_file");
    assert_eq!(call3_3.args["path"], "sklearn/impute/_iterative.py");

    // Pattern 4: Opening brace present, but key name omitted (`{":` or `{ ":`)
    let p4_1 =
        r#"{": "read_file", "path": "sklearn/impute/_iterative.py", "offset": 10, "limit": 50}"#;
    let call4_1 = parse_tool_call(p4_1).unwrap();
    assert_eq!(call4_1.name, "read_file");
    assert_eq!(call4_1.args["path"], "sklearn/impute/_iterative.py");
    assert_eq!(call4_1.args["offset"], 10);
    assert_eq!(call4_1.args["limit"], 50);

    let p4_2 = r#"{ ": "search", "pattern": "def _discover_files", "path": "pylint"}"#;
    let call4_2 = parse_tool_call(p4_2).unwrap();
    assert_eq!(call4_2.name, "search");
    assert_eq!(call4_2.args["pattern"], "def _discover_files");
    assert_eq!(call4_2.args["path"], "pylint");

    // Pattern 5: Preceding Chain-of-Thought / conversational reasoning
    let p5_1 = "Now I will read the target file to inspect the function:\n\": \"read_file\", \"path\": \"a.py\"}";
    let call5_1 = parse_tool_call(p5_1).unwrap();
    assert_eq!(call5_1.name, "read_file");
    assert_eq!(call5_1.args["path"], "a.py");

    let p5_2 = "Let me check git status:\ntool\": \"bash\", \"command\": \"git status\"}";
    let call5_2 = parse_tool_call(p5_2).unwrap();
    assert_eq!(call5_2.name, "bash");
    assert_eq!(call5_2.args["command"], "git status");

    // Pattern 6: Inside markdown codeblock
    let p6_1 = "```json\ntool\": \"bash\", \"command\": \"git status\"}\n```";
    let call6_1 = parse_tool_call(p6_1).unwrap();
    assert_eq!(call6_1.name, "bash");
    assert_eq!(call6_1.args["command"], "git status");

    // Direct cleaner repairs
    assert_eq!(
        clean_json_str(r#"tool": "bash", "command": "git diff"}"#),
        r#"{"tool": "bash", "command": "git diff"}"#
    );
    assert_eq!(
        clean_json_str(r#"": "read_file", "path": "a.py"}"#),
        r#"{"tool": "read_file", "path": "a.py"}"#
    );
    assert_eq!(
        clean_json_str(r#"{": "read_file", "path": "a.py"}"#),
        r#"{"tool": "read_file", "path": "a.py"}"#
    );
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
