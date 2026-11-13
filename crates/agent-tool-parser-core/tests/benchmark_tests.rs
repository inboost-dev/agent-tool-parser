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
fn test_unicode_multilingual_parameters() {
    let text = r#"<｜DSML｜tool_calls>
<｜DSML｜invoke name="recherche_documents">
<｜DSML｜parameter name="requête">rapport financier 2026</｜DSML｜parameter>
<｜DSML｜parameter name="dossier">documents/comptabilité</｜DSML｜parameter>
<｜DSML｜parameter name="mode_strict">true</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "recherche_documents");
    assert_eq!(call.args["requête"], "rapport financier 2026");
    assert_eq!(call.args["dossier"], "documents/comptabilité");
    assert_eq!(call.args["mode_strict"], "true");
}

#[test]
fn test_unicode_korean_hangul_parameters() {
    // 1. Korean JSON format
    let text_json = r#"{
        "tool": "파일_검색",
        "경로": "문서/보고서_2026.pdf",
        "옵션": {"정밀검색": true, "최대결과": 10}
    }"#;
    let call1 = parse_tool_call(text_json).unwrap();
    assert_eq!(call1.name, "파일_검색");
    assert_eq!(call1.args["경로"], "문서/보고서_2026.pdf");
    assert_eq!(call1.args["옵션"]["정밀검색"], true);
    assert_eq!(call1.args["옵션"]["최대결과"], 10);

    // 2. Korean XML format
    let text_xml = r#"<tool_use>
<name>코드_분석</name>
<arguments>
<프로젝트>인공지능_엔진</프로젝트>
<대상_파일>src/main.rs</대상_파일>
</arguments>
</tool_use>"#;
    let call2 = parse_tool_call(text_xml).unwrap();
    assert_eq!(call2.name, "코드_분석");
    assert_eq!(call2.args["프로젝트"], "인공지능_엔진");
    assert_eq!(call2.args["대상_파일"], "src/main.rs");

    // 3. Korean action prefix call
    let text_call =
        r#"호출: 데이터베이스_조회(테이블="사용자", 조건="활성_상태 == True", 제한=50)"#;
    let call3 = parse_tool_call(text_call).unwrap();
    assert_eq!(call3.name, "데이터베이스_조회");
    assert_eq!(call3.args["테이블"], "사용자");
    assert_eq!(call3.args["조건"], "활성_상태 == True");
    assert_eq!(call3.args["제한"], 50);
}

#[test]
fn test_unicode_indic_scripts_and_arabic() {
    // 1. Hindi (Devanagari) JSON
    let text_hindi_json = r#"{
        "tool": "फ़ाइल_खोज",
        "निर्देशिका": "दस्तावेज़/वित्तीय_रिपोर्ट",
        "सख्त_मोड": true
    }"#;
    let c1 = parse_tool_call(text_hindi_json).unwrap();
    assert_eq!(c1.name, "फ़ाइल_खोज");
    assert_eq!(c1.args["निर्देशिका"], "दस्तावेज़/वित्तीय_रिपोर्ट");
    assert_eq!(c1.args["सख्त_मोड"], true);

    // 2. Hindi (Devanagari) XML
    let text_hindi_xml = r#"<tool_use>
<name>डेटाबेस_क्वेरी</name>
<arguments>
<तालिका>उपयोगकर्ता</तालिका>
<सीमा>100</सीमा>
</arguments>
</tool_use>"#;
    let c2 = parse_tool_call(text_hindi_xml).unwrap();
    assert_eq!(c2.name, "डेटाबेस_क्वेरी");
    assert_eq!(c2.args["तालिका"], "उपयोगकर्ता");
    assert_eq!(c2.args["सीमा"], "100");

    // 3. Bengali (বাংলা) JSON
    let text_bengali = r#"{
        "tool": "নথি_অনুসন্ধান",
        "পথ": "প্রকল্প/উপাত্ত.csv",
        "অনুরোধ": "গ্রাহক_তালিকা"
    }"#;
    let c3 = parse_tool_call(text_bengali).unwrap();
    assert_eq!(c3.name, "নথি_অনুসন্ধান");
    assert_eq!(c3.args["পথ"], "প্রকল্প/উপাত্ত.csv");
    assert_eq!(c3.args["অনুরোধ"], "গ্রাহক_তালিকা");

    // 4. Tamil (தமிழ்) JSON
    let text_tamil = r#"{
        "tool": "தரவு_தேடல்",
        "கோப்பு": "அறிக்கை/2026.pdf",
        "வடிகட்டி": "முக்கியமானது"
    }"#;
    let c4 = parse_tool_call(text_tamil).unwrap();
    assert_eq!(c4.name, "தரவு_தேடல்");
    assert_eq!(c4.args["கோப்பு"], "அறிக்கை/2026.pdf");
    assert_eq!(c4.args["வடிகட்டி"], "முக்கியமானது");

    // 5. Telugu (తెలుగు) JSON
    let text_telugu = r#"{
        "tool": "ఫైల్_శోధన",
        "మార్గం": "నివేదిక/డేటా.csv",
        "పరిమితి": 50
    }"#;
    let c5 = parse_tool_call(text_telugu).unwrap();
    assert_eq!(c5.name, "ఫైల్_శోధన");
    assert_eq!(c5.args["మార్గం"], "నివేదిక/డేటా.csv");
    assert_eq!(c5.args["పరిమితి"], 50);

    // 6. Arabic (العربية - RTL script)
    let text_arabic = r#"{
        "tool": "بحث_الملفات",
        "مسار": "مستندات/تقرير_مالي.pdf",
        "خيارات": {"تطابق_تام": true}
    }"#;
    let c6 = parse_tool_call(text_arabic).unwrap();
    assert_eq!(c6.name, "بحث_الملفات");
    assert_eq!(c6.args["مسار"], "مستندات/تقرير_مالي.pdf");
    assert_eq!(c6.args["خيارات"]["تطابق_تام"], true);
}

#[test]
fn test_unicode_chinese_simplified_and_traditional() {
    // 1. Simplified Chinese in Qwen XML format
    let text_qwen = r#"<function=代码重构>
<parameter=目标文件>src/pipeline.py</parameter>
<parameter=重构策略>函数提取</parameter>
</function>"#;
    let call1 = parse_tool_call(text_qwen).unwrap();
    assert_eq!(call1.name, "代码重构");
    assert_eq!(call1.args["目标文件"], "src/pipeline.py");
    assert_eq!(call1.args["重构策略"], "函数提取");

    // 2. Traditional Chinese in JSON format
    let text_trad = r#"{
        "tool": "檔案檢索",
        "路徑": "系統設定/核心模組.json",
        "選項": {"除錯模式": true, "逾時秒數": 30}
    }"#;
    let call2 = parse_tool_call(text_trad).unwrap();
    assert_eq!(call2.name, "檔案檢索");
    assert_eq!(call2.args["路徑"], "系統設定/核心模組.json");
    assert_eq!(call2.args["選項"]["除錯模式"], true);
    assert_eq!(call2.args["選項"]["逾時秒數"], 30);

    // 3. Chinese prefix call
    let text_call = r#"调用: 数据分析(数据源="用户数据.csv", 输出格式="表格")"#;
    let call3 = parse_tool_call(text_call).unwrap();
    assert_eq!(call3.name, "数据分析");
    assert_eq!(call3.args["数据源"], "用户数据.csv");
    assert_eq!(call3.args["输出格式"], "表格");
}

#[test]
fn test_unicode_japanese_parameters() {
    // 1. Japanese JSON with Kanji, Katakana, and Hiragana
    let text_json = r#"{
        "tool": "ファイル検索",
        "ディレクトリ": "ドキュメント/設定ファイル",
        "キーワード": "データベース接続文字列",
        "再帰的検索": true
    }"#;
    let c1 = parse_tool_call(text_json).unwrap();
    assert_eq!(c1.name, "ファイル検索");
    assert_eq!(c1.args["ディレクトリ"], "ドキュメント/設定ファイル");
    assert_eq!(c1.args["キーワード"], "データベース接続文字列");
    assert_eq!(c1.args["再帰的検索"], true);

    // 2. Japanese in Qwen XML format
    let text_qwen = r#"<function=データベース_クエリ>
<parameter=テーブル名>ユーザー情報</parameter>
<parameter=抽出条件>有効ステータス</parameter>
</function>"#;
    let c2 = parse_tool_call(text_qwen).unwrap();
    assert_eq!(c2.name, "データベース_クエリ");
    assert_eq!(c2.args["テーブル名"], "ユーザー情報");
    assert_eq!(c2.args["抽出条件"], "有効ステータス");

    // 3. Japanese in Claude XML tag format
    let text_xml = r#"<tool_use>
<name>ファイルを読み込む</name>
<arguments>
<ファイルパス>設定/config.yaml</ファイルパス>
<エンコーディング>utf-8</エンコーディング>
</arguments>
</tool_use>"#;
    let c3 = parse_tool_call(text_xml).unwrap();
    assert_eq!(c3.name, "ファイルを読み込む");
    assert_eq!(c3.args["ファイルパス"], "設定/config.yaml");
    assert_eq!(c3.args["エンコーディング"], "utf-8");

    // 4. Japanese function call syntax
    let text_call =
        r#"テキスト解析(ソース="日本語の形態素解析テスト", モデル="gpt-4o", 詳細出力=True)"#;
    let c4 = parse_tool_call(text_call).unwrap();
    assert_eq!(c4.name, "テキスト解析");
    assert_eq!(c4.args["ソース"], "日本語の形態素解析テスト");
    assert_eq!(c4.args["モデル"], "gpt-4o");
    assert_eq!(c4.args["詳細出力"], true);
}

#[test]
fn test_unicode_cyrillic_parameters() {
    // 1. Cyrillic in DSML format
    let text_dsml = r#"<｜DSML｜tool_calls>
<｜DSML｜invoke name="поиск_документов">
<｜DSML｜parameter name="запрос">финансовый отчет 2026</｜DSML｜parameter>
<｜DSML｜parameter name="папка">документы/бухгалтерия</｜DSML｜parameter>
<｜DSML｜parameter name="строгий_режим">true</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"#;
    let call1 = parse_tool_call(text_dsml).unwrap();
    assert_eq!(call1.name, "поиск_документов");
    assert_eq!(call1.args["запрос"], "финансовый отчет 2026");
    assert_eq!(call1.args["папка"], "документы/бухгалтерия");
    assert_eq!(call1.args["строгий_режим"], "true");

    // 2. Cyrillic in XML format with CDATA
    let text_xml = r#"<tool_use>
<name>запись_файла</name>
<arguments>
<путь>src/модели.py</путь>
<содержимое><![CDATA[def получить_пользователя(): return "Иван"]]></содержимое>
</arguments>
</tool_use>"#;
    let call2 = parse_tool_call(text_xml).unwrap();
    assert_eq!(call2.name, "запись_файла");
    assert_eq!(call2.args["путь"], "src/модели.py");
    assert_eq!(
        call2.args["содержимое"],
        "def получить_пользователя(): return \"Иван\""
    );

    // 3. Russian action prefix call
    let text_call =
        r#"Вызов функции: поиск_информации(запрос="архитектура микросервисов", лимит=10)"#;
    let call3 = parse_tool_call(text_call).unwrap();
    assert_eq!(call3.name, "поиск_информации");
    assert_eq!(call3.args["запрос"], "архитектура микросервисов");
    assert_eq!(call3.args["лимит"], 10);
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
    let text = r#"{"tool": "logger", "text": "\u3053\u3093\u306b\u3061\u306f \u4e16\u754c"}"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "logger");
    assert_eq!(call.args["text"], "こんにちは 世界");
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

#[test]
fn test_qwen_xml_function_parameter_format() {
    let text = "<function=get_current_weather>\n    <parameter=location>San Francisco, CA</parameter>\n    <parameter=format>celsius</parameter>\n    </function>";
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "get_current_weather");
    assert_eq!(call.args["location"], "San Francisco, CA");
    assert_eq!(call.args["format"], "celsius");
}

#[test]
fn test_qwen_xml_wrapped_in_tool_call() {
    let text = r#"<tool_call>
<function=get_current_weather>
<parameter=location>Tokyo</parameter>
<parameter=unit>celsius</parameter>
</function>
</tool_call>"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "get_current_weather");
    assert_eq!(call.args["location"], "Tokyo");
    assert_eq!(call.args["unit"], "celsius");
}

#[test]
fn test_tool_call_with_raw_json() {
    let text = r#"<tool_call>
{"name": "fetch", "arguments": {"url": "https://api.inboost.pro"}}
</tool_call>"#;
    let call = parse_tool_call(text).unwrap();
    assert_eq!(call.name, "fetch");
    assert_eq!(call.args["url"], "https://api.inboost.pro");
}
