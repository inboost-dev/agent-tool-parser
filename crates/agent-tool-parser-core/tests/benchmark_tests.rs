// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT
//
// Clean-Room Synthesized Benchmark and Edge Case Test Suite for agent-tool-parser-core
//
// Provenance & IP Hygiene Notice:
// All test cases in this file are clean-room synthesized based on publicly observable
// function calling specifications and open industry benchmarks (e.g. Berkeley Function Calling
// Leaderboard / BFCL under Apache-2.0, DeepSeek R1/V3 technical reports, Anthropic Claude docs,
// OpenAI tool use specifications, Nous Hermes format). All synthetic inputs, assertion fixtures,
// and test structures are original code authored for agent-tool-parser under the MIT License.

use agent_tool_parser_core::{
    parse_tool_call, parse_tool_calls, try_parse_tool_call, try_parse_tool_calls,
};
use std::time::Instant;

#[test]
fn test_bfcl_simple_function_calling_json() {
    let text = r#"{
        "name": "get_weather",
        "arguments": {"location": "San Francisco, CA", "unit": "celsius"}
    }"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.args["location"], "San Francisco, CA");
    assert_eq!(call.args["unit"], "celsius");
}

#[test]
fn test_bfcl_simple_function_calling_anthropic() {
    let text = r#"<tool_use>
<name>get_weather</name>
<arguments>
{"location": "Tokyo, Japan", "unit": "celsius"}
</arguments>
</tool_use>"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.args["location"], "Tokyo, Japan");
    assert_eq!(call.args["unit"], "celsius");
}

#[test]
fn test_bfcl_simple_function_calling_dsml() {
    let text = r#"<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location">Berlin, Germany</｜DSML｜parameter>
<｜DSML｜parameter name="unit">celsius</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.args["location"], "Berlin, Germany");
    assert_eq!(call.args["unit"], "celsius");
}

#[test]
fn test_bfcl_heterogeneous_typed_arguments() {
    let text = r#"{
        "tool": "book_hotel",
        "hotel_name": "Grand Palace Hotel",
        "nights": 4,
        "rate_per_night": 189.95,
        "tax_exempt": false,
        "promo_code": null,
        "guest_names": ["Alice Smith", "Bob Smith"],
        "preferences": {
            "smoking": false,
            "floor": 12,
            "bed_type": "king"
        }
    }"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "book_hotel");
    assert_eq!(call.args["hotel_name"], "Grand Palace Hotel");
    assert_eq!(call.args["nights"], 4);
    assert_eq!(call.args["rate_per_night"], 189.95);
    assert_eq!(call.args["tax_exempt"], false);
    assert!(call.args["promo_code"].is_null());
    assert_eq!(call.args["guest_names"][0], "Alice Smith");
    assert_eq!(call.args["guest_names"][1], "Bob Smith");
    assert_eq!(call.args["preferences"]["floor"], 12);
    assert_eq!(call.args["preferences"]["smoking"], false);
}

#[test]
fn test_bfcl_parallel_function_calling_mixed() {
    let text = r#"[
        {"type": "function", "function": {"name": "get_stock_quote", "arguments": {"symbol": "AAPL"}}},
        {"type": "function", "function": {"name": "get_stock_quote", "arguments": {"symbol": "NVDA"}}},
        {"type": "function", "function": {"name": "get_market_news", "arguments": {"category": "tech", "limit": 5}}}
    ]"#;
    let calls = parse_tool_calls(text).unwrap();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0].name, "get_stock_quote");
    assert_eq!(calls[0].args["symbol"], "AAPL");
    assert_eq!(calls[1].name, "get_stock_quote");
    assert_eq!(calls[1].args["symbol"], "NVDA");
    assert_eq!(calls[2].name, "get_market_news");
    assert_eq!(calls[2].args["category"], "tech");
    assert_eq!(calls[2].args["limit"], 5);
}

#[test]
fn test_bfcl_conversational_rejection() {
    let conversations = [
        "Hello! I am a helpful AI assistant. How may I help you today?",
        "To solve this problem, you should consider using dynamic programming.\nHere is an example in Python:\ndef solve(): pass",
        "I checked the information and it appears that today's temperature is 22 degrees Celsius.",
    ];
    for conv in conversations {
        assert_eq!(try_parse_tool_calls(conv), Vec::new());
        assert!(try_parse_tool_call(conv).is_none());
        assert!(parse_tool_call(conv).is_err());
    }
}

#[test]
fn test_unicode_cyrillic_parameters() {
    let text = r#"<｜DSML｜tool_calls>
<｜DSML｜invoke name="поиск_документов">
<｜DSML｜parameter name="запрос">финансовый отчет 2026</｜DSML｜parameter>
<｜DSML｜parameter name="папка">документы/бухгалтерия</｜DSML｜parameter>
<｜DSML｜parameter name="строгий_режим">true</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "поиск_документов");
    assert_eq!(call.args["запрос"], "финансовый отчет 2026");
    assert_eq!(call.args["папка"], "документы/бухгалтерия");
    assert_eq!(call.args["строгий_режим"], "true");
}

#[test]
fn test_unicode_cjk_parameters() {
    let text = r#"{
        "tool": "file_search",
        "query": "人工智能深度学习架构",
        "directory": "研究论文/2026年"
    }"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "file_search");
    // "query" is normalized via default parameter aliases to "pattern"
    assert_eq!(call.args["pattern"], "人工智能深度学习架构");
    assert_eq!(call.args["directory"], "研究论文/2026年");
}

#[test]
fn test_unicode_emoji_and_special_symbols() {
    let text = r#"<tool_call>
{"name": "send_notification", "arguments": {"message": "Release v1.0.0 is live! 🚀✨\nStatus: Verified ✅", "math_symbol": "∀x ∈ ℝ: x² ≥ 0"}}
</tool_call>"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "send_notification");
    let msg = call.args["message"].as_str().unwrap();
    assert!(msg.contains("🚀✨"));
    assert!(msg.contains("✅"));
    assert_eq!(call.args["math_symbol"], "∀x ∈ ℝ: x² ≥ 0");
}

#[test]
fn test_unicode_escaped_sequences() {
    let text =
        r#"{"tool": "logger", "text": "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"}"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "logger");
    assert_eq!(call.args["text"], "Привет мир");
}

#[test]
fn test_dirty_json_trailing_commas_in_objects_and_arrays() {
    let text = r#"{"tool": "bash", "command": "cargo check", "flags": ["--all", "--release", ], }"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    assert_eq!(call.args["command"], "cargo check");
    assert_eq!(call.args["flags"][0], "--all");
    assert_eq!(call.args["flags"][1], "--release");
}

#[test]
fn test_dirty_json_single_quotes_and_python_constants() {
    let text = r#"{'tool': 'sync_db', 'dry_run': True, 'verbose': False, 'batch_size': None, 'tag': 'v1'}"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "sync_db");
    assert_eq!(call.args["dry_run"], true);
    assert_eq!(call.args["verbose"], false);
    assert!(call.args["batch_size"].is_null());
    assert_eq!(call.args["tag"], "v1");
}

#[test]
fn test_cdata_nested_xml_tags_isolation() {
    let xml_payload = "<widget id=\"42\">\n    <item name=\"inner_item\">Hello World</item>\n    <parameter name=\"fake_param\">Do not parse me</parameter>\n</widget>";
    let text = format!(
        "<｜DSML｜invoke name=\"save_template\">\n<｜DSML｜parameter name=\"template_name\">template.xml</｜DSML｜parameter>\n<｜DSML｜parameter name=\"content\"><![CDATA[\n{}\n]]></｜DSML｜parameter>\n</｜DSML｜invoke>",
        xml_payload
    );
    let call = parse_tool_call(&text).unwrap();
    assert_eq!(call.name, "save_template");
    assert_eq!(call.args["template_name"], "template.xml");
    assert_eq!(call.args["content"].as_str().unwrap().trim(), xml_payload);
    assert!(call.args.get("fake_param").is_none());
}

#[test]
fn test_dsml_attribute_quoting_variants() {
    let unquoted = "<｜DSML｜tool name=search><｜DSML｜parameter name=query string=true>deepseek</｜DSML｜parameter></｜DSML｜tool>";
    let call_unquoted = parse_tool_call(unquoted).unwrap();
    assert_eq!(call_unquoted.name, "search");
    assert_eq!(call_unquoted.args["pattern"], "deepseek");

    let single_quoted = "<｜DSML｜tool name='search'><｜DSML｜parameter name='query'>llama</｜DSML｜parameter></｜DSML｜tool>";
    let call_single = parse_tool_call(single_quoted).unwrap();
    assert_eq!(call_single.name, "search");
    assert_eq!(call_single.args["pattern"], "llama");
}

#[test]
fn test_streaming_truncated_mid_string() {
    let text = "<｜tool calls begin｜><｜tool call begin｜>function=execute_script<｜tool sep｜>{\"command\": \"python -m build";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "execute_script");
    let cmd = call.args["command"].as_str().unwrap();
    assert!(cmd.starts_with("python -m build"));
}

#[test]
fn test_react_crlf_and_multiline_script() {
    let text = "Thought: I need to update submodules.\r\nAction: bash\r\nAction Input: git submodule sync\r\ngit submodule update --init --recursive\r\n";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "bash");
    let cmd = call.args["command"].as_str().unwrap();
    assert!(cmd.contains("git submodule sync"));
    assert!(cmd.contains("git submodule update"));
}

#[test]
fn test_python_ast_complex_expressions() {
    let text = "deploy_service(\"staging\", replicas=3, env_vars=[\"FOO=bar\", \"BAZ=qux\"], dry_run=True)";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "deploy_service");
    assert_eq!(call.args["arg0"], "staging");
    assert_eq!(call.args["replicas"], 3);
    assert_eq!(call.args["dry_run"], true);
}

#[test]
fn test_empty_and_whitespace_rejection() {
    let cases = ["", "   ", "\t\t\n", "\r\n  \r\n"];
    for empty in cases {
        assert_eq!(try_parse_tool_calls(empty), Vec::new());
        assert!(try_parse_tool_call(empty).is_none());
        assert!(parse_tool_call(empty).is_err());
    }
}

#[test]
fn test_large_payload_stability_and_performance() {
    let synthetic_lines: Vec<String> = (0..500)
        .map(|i| format!("    let line_{} = calculate_value({});", i, i))
        .collect();
    let synthetic_code = synthetic_lines.join("\n");
    let text = format!(
        "<invoke name=\"write_file\">\n<parameter name=\"path\">crates/large_generated.rs</parameter>\n<parameter name=\"content\"><![CDATA[\n{}\n]]></parameter>\n</invoke>",
        synthetic_code
    );

    let start = Instant::now();
    let call = parse_tool_call(&text).unwrap();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(call.name, "write_file");
    assert_eq!(call.args["path"], "crates/large_generated.rs");
    let content = call.args["content"].as_str().unwrap();
    assert!(content.contains("line_499"));
    assert!(
        elapsed_ms < 1500.0,
        "Parsing took too long: {} ms",
        elapsed_ms
    );
}
