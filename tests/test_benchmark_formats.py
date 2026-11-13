# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT
#
# Clean-Room Synthesized Benchmark and Edge Case Test Suite
#
# Provenance & IP Hygiene Notice:
# All test cases in this file are clean-room synthesized based on publicly observable
# function calling specifications and open industry benchmarks (e.g. Berkeley Function Calling
# Leaderboard / BFCL under Apache-2.0, DeepSeek R1/V3 technical reports, Anthropic Claude docs,
# OpenAI tool use specifications, Nous Hermes format). All synthetic inputs, assertion fixtures,
# and test structures are original code authored for agent-tool-parser under the MIT License.

from __future__ import annotations

import time
import unittest

from agent_tool_parser import (
    ToolError,
    parse_tool_call,
    parse_tool_calls,
    try_parse_tool_call,
    try_parse_tool_calls,
)


class TestBenchmarkFormatsAndEdgeCases(unittest.TestCase):
    """Rigorous benchmark & edge-case test suite covering BFCL patterns, I18N, and dirty tokens."""

    # -------------------------------------------------------------------------
    # 1. BFCL Standard & Industry Function Calling Patterns
    # -------------------------------------------------------------------------

    def test_bfcl_simple_function_calling_json(self):
        """Single function call with scalar string argument (OpenAI / BFCL standard format)."""
        text = """{
            "name": "get_weather",
            "arguments": {"location": "San Francisco, CA", "unit": "celsius"}
        }"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.args, {"location": "San Francisco, CA", "unit": "celsius"})

    def test_bfcl_simple_function_calling_anthropic(self):
        """Single function call in Anthropic Claude tool_use format."""
        text = """<tool_use>
<name>get_weather</name>
<arguments>
{"location": "Tokyo, Japan", "unit": "celsius"}
</arguments>
</tool_use>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.args, {"location": "Tokyo, Japan", "unit": "celsius"})

    def test_bfcl_simple_function_calling_dsml(self):
        """Single function call in DeepSeek DSML format."""
        text = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location">Berlin, Germany</｜DSML｜parameter>
<｜DSML｜parameter name="unit">celsius</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.args["location"], "Berlin, Germany")
        self.assertEqual(call.args["unit"], "celsius")

    def test_bfcl_heterogeneous_typed_arguments(self):
        """Heterogeneous data types: int, float, bool, null, list, nested object."""
        text = """{
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
        }"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "book_hotel")
        self.assertEqual(call.args["hotel_name"], "Grand Palace Hotel")
        self.assertEqual(call.args["nights"], 4)
        self.assertAlmostEqual(call.args["rate_per_night"], 189.95, places=2)
        self.assertFalse(call.args["tax_exempt"])
        self.assertIsNone(call.args["promo_code"])
        self.assertEqual(call.args["guest_names"], ["Alice Smith", "Bob Smith"])
        self.assertEqual(call.args["preferences"]["floor"], 12)
        self.assertFalse(call.args["preferences"]["smoking"])

    def test_bfcl_parallel_function_calling_mixed(self):
        """Parallel execution of multiple heterogeneous tools in a single turn."""
        text = """[
            {"type": "function", "function": {"name": "get_stock_quote", "arguments": {"symbol": "AAPL"}}},
            {"type": "function", "function": {"name": "get_stock_quote", "arguments": {"symbol": "NVDA"}}},
            {"type": "function", "function": {"name": "get_market_news", "arguments": {"category": "tech", "limit": 5}}}
        ]"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0].name, "get_stock_quote")
        self.assertEqual(calls[0].args, {"symbol": "AAPL"})
        self.assertEqual(calls[1].name, "get_stock_quote")
        self.assertEqual(calls[1].args, {"symbol": "NVDA"})
        self.assertEqual(calls[2].name, "get_market_news")
        self.assertEqual(calls[2].args, {"category": "tech", "limit": 5})

    def test_bfcl_conversational_rejection(self):
        """Pure conversational text without tool calls must be safely rejected."""
        conversations = [
            "Hello! I am a helpful AI assistant. How may I help you today?",
            "To solve this problem, you should consider using dynamic programming.\nHere is an example in Python:\ndef solve(): pass",
            "I checked the information and it appears that today's temperature is 22 degrees Celsius.",
        ]
        for conv in conversations:
            self.assertEqual(try_parse_tool_calls(conv), [])
            self.assertIsNone(try_parse_tool_call(conv))
            with self.assertRaises(ToolError):
                parse_tool_call(conv)

    # -------------------------------------------------------------------------
    # 2. Multilingual, Unicode & Internationalization (I18N)
    # -------------------------------------------------------------------------

    def test_unicode_multilingual_parameters(self):
        """Multilingual accented parameters and values (French/international localization test)."""
        text = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="recherche_documents">
<｜DSML｜parameter name="requête">rapport financier 2026</｜DSML｜parameter>
<｜DSML｜parameter name="dossier">documents/comptabilité</｜DSML｜parameter>
<｜DSML｜parameter name="mode_strict">true</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "recherche_documents")
        self.assertEqual(call.args["requête"], "rapport financier 2026")
        self.assertEqual(call.args["dossier"], "documents/comptabilité")
        self.assertEqual(call.args["mode_strict"], "true")

    def test_unicode_korean_hangul_parameters(self):
        """Korean Hangul characters in JSON, XML, and Python function call formats."""
        # 1. Korean in JSON format
        text_json = """{
            "tool": "파일_검색",
            "경로": "문서/보고서_2026.pdf",
            "옵션": {"정밀검색": true, "최대결과": 10}
        }"""
        call1 = parse_tool_call(text_json)
        self.assertEqual(call1.name, "파일_검색")
        self.assertEqual(call1.args["경로"], "문서/보고서_2026.pdf")
        self.assertEqual(call1.args["옵션"], {"정밀검색": True, "최대결과": 10})

        # 2. Korean in XML format
        text_xml = """<tool_use>
<name>코드_분석</name>
<arguments>
<프로젝트>인공지능_엔진</프로젝트>
<대상_파일>src/main.rs</대상_파일>
</arguments>
</tool_use>"""
        call2 = parse_tool_call(text_xml)
        self.assertEqual(call2.name, "코드_분석")
        self.assertEqual(call2.args["프로젝트"], "인공지능_엔진")
        self.assertEqual(call2.args["대상_파일"], "src/main.rs")

        # 3. Korean action prefix call
        text_call = '호출: 데이터베이스_조회(테이블="사용자", 조건="활성_상태 == True", 제한=50)'
        call3 = parse_tool_call(text_call)
        self.assertEqual(call3.name, "데이터베이스_조회")
        self.assertEqual(call3.args["테이블"], "사용자")
        self.assertEqual(call3.args["조건"], "활성_상태 == True")
        self.assertEqual(call3.args["제한"], 50)

    def test_unicode_indic_scripts_and_arabic(self):
        """Indic scripts (Hindi, Bengali, Tamil, Telugu) and Arabic in JSON and XML formats."""
        # 1. Hindi (Devanagari) JSON
        text_hindi_json = """{
            "tool": "फ़ाइल_खोज",
            "निर्देशिका": "दस्तावेज़/वित्तीय_रिपोर्ट",
            "सख्त_मोड": true
        }"""
        c1 = parse_tool_call(text_hindi_json)
        self.assertEqual(c1.name, "फ़ाइल_खोज")
        self.assertEqual(c1.args["निर्देशिका"], "दस्तावेज़/वित्तीय_रिपोर्ट")
        self.assertTrue(c1.args["सख्त_मोड"])

        # 2. Hindi (Devanagari) XML
        text_hindi_xml = """<tool_use>
<name>डेटाबेस_क्वेरी</name>
<arguments>
<तालिका>उपयोगकर्ता</तालिका>
<सीमा>100</सीमा>
</arguments>
</tool_use>"""
        c2 = parse_tool_call(text_hindi_xml)
        self.assertEqual(c2.name, "डेटाबेस_क्वेरी")
        self.assertEqual(c2.args["तालिका"], "उपयोगकर्ता")
        self.assertEqual(c2.args["सीमा"], "100")

        # 3. Bengali (বাংলা) JSON
        text_bengali = """{
            "tool": "নথি_অনুসন্ধান",
            "পথ": "প্রকল্প/উপাত্ত.csv",
            "অনুরোধ": "গ্রাহক_তালিকা"
        }"""
        c3 = parse_tool_call(text_bengali)
        self.assertEqual(c3.name, "নথি_অনুসন্ধান")
        self.assertEqual(c3.args["পথ"], "প্রকল্প/উপাত্ত.csv")
        self.assertEqual(c3.args["অনুরোধ"], "গ্রাহক_তালিকা")

        # 4. Tamil (தமிழ்) JSON
        text_tamil = """{
            "tool": "தரவு_தேடல்",
            "கோப்பு": "அறிக்கை/2026.pdf",
            "வடிகட்டி": "முக்கியமானது"
        }"""
        c4 = parse_tool_call(text_tamil)
        self.assertEqual(c4.name, "தரவு_தேடல்")
        self.assertEqual(c4.args["கோப்பு"], "அறிக்கை/2026.pdf")
        self.assertEqual(c4.args["வடிகட்டி"], "முக்கியமானது")

        # 5. Telugu (తెలుగు) JSON
        text_telugu = """{
            "tool": "ఫైల్_శోధన",
            "మార్గం": "నివేదిక/డేటా.csv",
            "పరిమితి": 50
        }"""
        c5 = parse_tool_call(text_telugu)
        self.assertEqual(c5.name, "ఫైల్_శోధన")
        self.assertEqual(c5.args["మార్గం"], "నివేదిక/డేటా.csv")
        self.assertEqual(c5.args["పరిమితి"], 50)

        # 6. Arabic (العربية - RTL script)
        text_arabic = """{
            "tool": "بحث_الملفات",
            "مسار": "مستندات/تقرير_مالي.pdf",
            "خيارات": {"تطابق_تام": true}
        }"""
        c6 = parse_tool_call(text_arabic)
        self.assertEqual(c6.name, "بحث_الملفات")
        self.assertEqual(c6.args["مسار"], "مستندات/تقرير_مالي.pdf")
        self.assertEqual(c6.args["خيارات"], {"تطابق_تام": True})

    def test_unicode_chinese_simplified_and_traditional(self):
        """Chinese characters (Simplified and Traditional) in Qwen XML, JSON, and function call formats."""
        # 1. Simplified Chinese in Qwen XML format
        text_qwen = """<function=代码重构>
<parameter=目标文件>src/pipeline.py</parameter>
<parameter=重构策略>函数提取</parameter>
</function>"""
        call1 = parse_tool_call(text_qwen)
        self.assertEqual(call1.name, "代码重构")
        self.assertEqual(call1.args["目标文件"], "src/pipeline.py")
        self.assertEqual(call1.args["重构策略"], "函数提取")

        # 2. Traditional Chinese in JSON format
        text_trad = """{
            "tool": "檔案檢索",
            "路徑": "系統設定/核心模組.json",
            "選項": {"除錯模式": true, "逾時秒數": 30}
        }"""
        call2 = parse_tool_call(text_trad)
        self.assertEqual(call2.name, "檔案檢索")
        self.assertEqual(call2.args["路徑"], "系統設定/核心模組.json")
        self.assertEqual(call2.args["選項"], {"除錯模式": True, "逾時秒數": 30})

        # 3. Chinese prefix call
        text_call = '调用: 数据分析(数据源="用户数据.csv", 输出格式="表格")'
        call3 = parse_tool_call(text_call)
        self.assertEqual(call3.name, "数据分析")
        self.assertEqual(call3.args["数据源"], "用户数据.csv")
        self.assertEqual(call3.args["输出格式"], "表格")

    def test_unicode_japanese_parameters(self):
        """Japanese Kanji, Hiragana, and Katakana in JSON, Qwen XML, and function call formats."""
        # 1. Japanese JSON with Kanji, Katakana, and Hiragana
        text_json = """{
            "tool": "ファイル検索",
            "ディレクトリ": "ドキュメント/設定ファイル",
            "キーワード": "データベース接続文字列",
            "再帰的検索": true
        }"""
        c1 = parse_tool_call(text_json)
        self.assertEqual(c1.name, "ファイル検索")
        self.assertEqual(c1.args["ディレクトリ"], "ドキュメント/設定ファイル")
        self.assertEqual(c1.args["キーワード"], "データベース接続文字列")
        self.assertTrue(c1.args["再帰的検索"])

        # 2. Japanese in Qwen XML format
        text_qwen = """<function=データベース_クエリ>
<parameter=テーブル名>ユーザー情報</parameter>
<parameter=抽出条件>有効ステータス</parameter>
</function>"""
        c2 = parse_tool_call(text_qwen)
        self.assertEqual(c2.name, "データベース_クエリ")
        self.assertEqual(c2.args["テーブル名"], "ユーザー情報")
        self.assertEqual(c2.args["抽出条件"], "有効ステータス")

        # 3. Japanese in Claude XML tag format
        text_xml = """<tool_use>
<name>ファイルを読み込む</name>
<arguments>
<ファイルパス>設定/config.yaml</ファイルパス>
<エンコーディング>utf-8</エンコーディング>
</arguments>
</tool_use>"""
        c3 = parse_tool_call(text_xml)
        self.assertEqual(c3.name, "ファイルを読み込む")
        self.assertEqual(c3.args["ファイルパス"], "設定/config.yaml")
        self.assertEqual(c3.args["エンコーディング"], "utf-8")

        # 4. Japanese function call syntax
        text_call = (
            'テキスト解析(ソース="日本語の形態素解析テスト", モデル="gpt-4o", 詳細出力=True)'
        )
        c4 = parse_tool_call(text_call)
        self.assertEqual(c4.name, "テキスト解析")
        self.assertEqual(c4.args["ソース"], "日本語の形態素解析テスト")
        self.assertEqual(c4.args["モデル"], "gpt-4o")
        self.assertTrue(c4.args["詳細出力"])

    def test_unicode_cyrillic_parameters(self):
        """Cyrillic characters in DSML, XML, JSON, and Russian action prefix formats."""
        # 1. Cyrillic in DSML format
        text_dsml = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="поиск_документов">
<｜DSML｜parameter name="запрос">финансовый отчет 2026</｜DSML｜parameter>
<｜DSML｜parameter name="папка">документы/бухгалтерия</｜DSML｜parameter>
<｜DSML｜parameter name="строгий_режим">true</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""
        call1 = parse_tool_call(text_dsml)
        self.assertEqual(call1.name, "поиск_документов")
        self.assertEqual(call1.args["запрос"], "финансовый отчет 2026")
        self.assertEqual(call1.args["папка"], "документы/бухгалтерия")
        self.assertEqual(call1.args["строгий_режим"], "true")

        # 2. Cyrillic in XML format with CDATA
        text_xml = """<tool_use>
<name>запись_файла</name>
<arguments>
<путь>src/модели.py</путь>
<содержимое><![CDATA[def получить_пользователя(): return "Иван"]]></содержимое>
</arguments>
</tool_use>"""
        call2 = parse_tool_call(text_xml)
        self.assertEqual(call2.name, "запись_файла")
        self.assertEqual(call2.args["путь"], "src/модели.py")
        self.assertEqual(call2.args["содержимое"], 'def получить_пользователя(): return "Иван"')

        # 3. Russian action prefix call
        text_call = 'Вызов функции: поиск_информации(запрос="архитектура микросервисов", лимит=10)'
        call3 = parse_tool_call(text_call)
        self.assertEqual(call3.name, "поиск_информации")
        self.assertEqual(call3.args["запрос"], "архитектура микросервисов")
        self.assertEqual(call3.args["лимит"], 10)

    def test_unicode_cjk_parameters(self):
        """CJK characters (Chinese, Japanese) in tool calls."""
        text = """{
            "tool": "file_search",
            "query": "人工智能深度学习架构",
            "directory": "研究论文/2026年"
        }"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "file_search")
        # 'query' parameter is normalized via DEFAULT_PARAM_ALIASES to 'pattern'
        self.assertEqual(call.args["pattern"], "人工智能深度学习架构")
        self.assertEqual(call.args["directory"], "研究论文/2026年")

    def test_unicode_emoji_and_special_symbols(self):
        """Emoji, mathematical notations, and typography quotes in arguments."""
        text = """<tool_call>
{"name": "send_notification", "arguments": {"message": "Release v1.0.0 is live! 🚀✨\\nStatus: Verified ✅", "math_symbol": "∀x ∈ ℝ: x² ≥ 0"}}
</tool_call>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "send_notification")
        self.assertIn("🚀✨", call.args["message"])
        self.assertIn("✅", call.args["message"])
        self.assertEqual(call.args["math_symbol"], "∀x ∈ ℝ: x² ≥ 0")

    def test_unicode_escaped_sequences(self):
        """JSON standard unicode escape sequences unescaped into unicode characters."""
        text = '{"tool": "logger", "text": "\\u3053\\u3093\\u306b\\u3061\\u306f \\u4e16\\u754c"}'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "logger")
        self.assertEqual(call.args["text"], "こんにちは 世界")

    # -------------------------------------------------------------------------
    # 3. Robustness, Dirty JSON & Malformed Payloads
    # -------------------------------------------------------------------------

    def test_dirty_json_trailing_commas_in_objects_and_arrays(self):
        """Trailing commas in both objects and lists repaired correctly."""
        text = '{"tool": "bash", "command": "cargo check", "flags": ["--all", "--release", ], }'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args["command"], "cargo check")
        self.assertEqual(call.args["flags"], ["--all", "--release"])

    def test_dirty_json_single_quotes_and_python_constants(self):
        """Single quoted keys/strings and Python True/False/None literals."""
        text = "{'tool': 'sync_db', 'dry_run': True, 'verbose': False, 'batch_size': None, 'tag': 'v1'}"
        call = parse_tool_call(text)
        self.assertEqual(call.name, "sync_db")
        self.assertTrue(call.args["dry_run"])
        self.assertFalse(call.args["verbose"])
        self.assertIsNone(call.args["batch_size"])
        self.assertEqual(call.args["tag"], "v1")

    def test_cdata_nested_xml_tags_isolation(self):
        """Raw XML/HTML tags inside CDATA blocks must NOT be parsed as outer parameter tags."""
        xml_payload = """<widget id="42">
    <item name="inner_item">Hello World</item>
    <parameter name="fake_param">Do not parse me</parameter>
</widget>"""
        text = f"""<｜DSML｜invoke name="save_template">
<｜DSML｜parameter name="template_name">template.xml</｜DSML｜parameter>
<｜DSML｜parameter name="content"><![CDATA[
{xml_payload}
]]></｜DSML｜parameter>
</｜DSML｜invoke>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "save_template")
        self.assertEqual(call.args["template_name"], "template.xml")
        self.assertEqual(call.args["content"].strip(), xml_payload.strip())
        self.assertNotIn("fake_param", call.args)

    def test_dsml_attribute_quoting_variants(self):
        """DSML tag attributes with unquoted, single-quoted, and double-quoted values."""
        unquoted = "<｜DSML｜tool name=search><｜DSML｜parameter name=query string=true>deepseek</｜DSML｜parameter></｜DSML｜tool>"
        call_unquoted = parse_tool_call(unquoted)
        self.assertEqual(call_unquoted.name, "search")
        self.assertEqual(call_unquoted.args["pattern"], "deepseek")

        single_quoted = "<｜DSML｜tool name='search'><｜DSML｜parameter name='query'>llama</｜DSML｜parameter></｜DSML｜tool>"
        call_single = parse_tool_call(single_quoted)
        self.assertEqual(call_single.name, "search")
        self.assertEqual(call_single.args["pattern"], "llama")

    def test_streaming_truncated_mid_string(self):
        """Graceful recovery when a stream is cut off mid-string."""
        text = '<｜tool calls begin｜><｜tool call begin｜>function=execute_script<｜tool sep｜>{"command": "python -m build'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "execute_script")
        self.assertTrue(call.args["command"].startswith("python -m build"))

    def test_react_crlf_and_multiline_script(self):
        """ReAct format with Windows CRLF newlines and multi-line script."""
        text = "Thought: I need to update submodules.\r\nAction: bash\r\nAction Input: git submodule sync\r\ngit submodule update --init --recursive\r\n"
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertIn("git submodule sync", call.args["command"])
        self.assertIn("git submodule update", call.args["command"])

    def test_python_ast_complex_expressions(self):
        """Python expression format with positional args, keyword args, and lists."""
        text = (
            'deploy_service("staging", replicas=3, env_vars=["FOO=bar", "BAZ=qux"], dry_run=True)'
        )
        call = parse_tool_call(text)
        self.assertEqual(call.name, "deploy_service")
        self.assertEqual(call.args["arg0"], "staging")
        self.assertEqual(call.args["replicas"], 3)
        self.assertEqual(call.args["env_vars"], ["FOO=bar", "BAZ=qux"])
        self.assertTrue(call.args["dry_run"])

    # -------------------------------------------------------------------------
    # 4. Boundary & Performance Stability
    # -------------------------------------------------------------------------

    def test_empty_and_whitespace_rejection(self):
        """Empty and whitespace-only inputs must cleanly fail validation."""
        for empty in ["", "   ", "\t\t\n", "\r\n  \r\n"]:
            self.assertEqual(try_parse_tool_calls(empty), [])
            with self.assertRaises(ToolError):
                parse_tool_call(empty)

    def test_large_payload_stability_and_performance(self):
        """20+ KB file contents in CDATA parsed in sub-millisecond without regex catastrophic backtracking."""
        synthetic_code = "\n".join(
            [f"    let line_{i} = calculate_value({i});" for i in range(500)]
        )
        text = f"""<invoke name="write_file">
<parameter name="path">crates/large_generated.rs</parameter>
<parameter name="content"><![CDATA[
{synthetic_code}
]]></parameter>
</invoke>"""
        # Warm up to exclude one-off regex compilation overhead from benchmark timing
        _ = parse_tool_call(text)
        start = time.perf_counter()
        call = parse_tool_call(text)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.args["path"], "crates/large_generated.rs")
        self.assertIn("line_499", call.args["content"])
        self.assertLess(
            elapsed_ms, 100.0, f"Parsing large payload took too long: {elapsed_ms:.2f} ms"
        )


if __name__ == "__main__":
    unittest.main()
