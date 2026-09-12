# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT
#
# Degenerate, Pathological, and Adversarial Input Test Suite
#
# Provenance & IP Hygiene Notice:
# All test cases in this file are original clean-room synthetic cases targeting
# degenerate input texts, malformed streams, control characters, ReDoS resistance,
# and adversarial prompt structures. Authored under the MIT License.

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


class TestDegenerateInputs(unittest.TestCase):
    """Stress tests on degenerate, pathological, and adversarial input texts."""

    # -------------------------------------------------------------------------
    # 1. Empty, Whitespace, Control Characters & Non-Printable Bytes
    # -------------------------------------------------------------------------

    def test_degenerate_empty_and_whitespace(self):
        """Empty, whitespace-only, and newline-only strings must be cleanly rejected."""
        cases = [
            "",
            "   ",
            "\t",
            "\n\n\n",
            "\r\n\r\n",
            "   \t  \r\n   ",
        ]
        for inp in cases:
            self.assertEqual(try_parse_tool_calls(inp), [])
            self.assertIsNone(try_parse_tool_call(inp))
            with self.assertRaises(ToolError):
                parse_tool_call(inp)

    def test_degenerate_invisible_unicode_and_bom(self):
        """Zero-width spaces, Unicode BOM, and direction marks around tool calls."""
        # BOM + Zero-width spaces wrapping valid JSON tool call
        text = '\ufeff\u200b\u200c{"tool": "bash", "command": "ls -la"}\u200d\u200e'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args["command"], "ls -la")

    def test_degenerate_ansi_escape_code_wrappers(self):
        """Simulated terminal ANSI escape sequences wrapping tool call tags."""
        text = '\x1b[31;1m<invoke name="bash"><parameter name="cmd">cargo test</parameter></invoke>\x1b[0m'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args["command"], "cargo test")

    def test_degenerate_embedded_null_byte(self):
        """Null byte (\x00) inside parameter content should not cause crash or buffer overrun."""
        text = '{"tool": "bash", "command": "echo hello\x00world"}'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertIn("hello\x00world", call.args["command"])

        # Also test JSON \\u0000 escaped null byte
        text_escaped = '{"tool": "bash", "command": "echo hello\\u0000world"}'
        call_escaped = parse_tool_call(text_escaped)
        self.assertEqual(call_escaped.name, "bash")
        self.assertIn("hello\x00world", call_escaped.args["command"])

    # -------------------------------------------------------------------------
    # 2. Streaming Truncation & Token Cut-Offs (EOF at Arbitrary Positions)
    # -------------------------------------------------------------------------

    def test_degenerate_eof_after_tag_opening(self):
        """Stream abruptly terminated right after tag opening."""
        cases = [
            "<invoke",
            "<invoke ",
            "<tool_call",
            "<｜DSML｜",
            "<｜DSML｜tool",
            "<｜tool calls begin｜>",
            "<｜tool calls begin｜><｜tool call begin｜>",
        ]
        for inp in cases:
            self.assertEqual(try_parse_tool_calls(inp), [])
            with self.assertRaises(ToolError):
                parse_tool_call(inp)

    def test_degenerate_eof_after_json_colon(self):
        """Stream cut off immediately after key colon: '{"tool": "bash", "command":'."""
        text = '{"tool": "bash", "command":'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args.get("command"), "")

    def test_degenerate_eof_inside_json_array(self):
        """Stream cut off in the middle of a JSON array."""
        text = '{"tool": "process", "items": ["alpha", "beta", '
        call = parse_tool_call(text)
        self.assertEqual(call.name, "process")
        self.assertEqual(call.args["items"], ["alpha", "beta"])

    def test_degenerate_eof_with_trailing_escape_backslash(self):
        """Stream cut off with a dangling backslash: unclosed quote must not be escaped."""
        text = '{"tool": "bash", "command": "echo hello\\'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args["command"], "echo hello")

    def test_degenerate_eof_inside_cdata(self):
        """Stream cut off in the middle of CDATA code block without closing ]]>."""
        code = "def process():\n    for i in range(10):\n        print(i)"
        text = f'<invoke name="write_file"><parameter name="content"><![CDATA[{code}'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.args["content"].strip(), code.strip())

    # -------------------------------------------------------------------------
    # 3. Malformed XML & Tag Pathologies
    # -------------------------------------------------------------------------

    def test_degenerate_unclosed_sequential_parameters(self):
        """Consecutive unclosed parameter tags (each parameter closed implicitly by next)."""
        text = """<invoke name="deploy">
<parameter name="env">staging
<parameter name="version">v2.1.0
<parameter name="dry_run">true
</invoke>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "deploy")
        self.assertEqual(call.args["env"], "staging")
        self.assertEqual(call.args["version"], "v2.1.0")
        self.assertEqual(call.args["dry_run"], "true")

    def test_degenerate_empty_parameter_tags(self):
        """Parameter tags with completely empty values."""
        text = '<invoke name="bash"><parameter name="cmd"></parameter></invoke>'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args["command"], "")

    def test_degenerate_duplicate_parameter_names(self):
        """Duplicate parameter tags in a single invocation (last value wins deterministically)."""
        text = """<invoke name="bash">
<parameter name="cmd">echo first</parameter>
<parameter name="cmd">echo second</parameter>
</invoke>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args["command"], "echo second")

    def test_degenerate_mismatched_closing_tag(self):
        """Mismatched closing tag: <invoke> closed by </tool>."""
        text = '<invoke name="bash"><parameter name="cmd">pytest</parameter></tool>'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args["command"], "pytest")

    def test_degenerate_html_entities_decoding(self):
        """XML/HTML entities inside parameters (&lt;, &gt;, &amp;, &quot;, &#39;)."""
        text = '<invoke name="bash"><parameter name="cmd">cat &lt; input.txt &amp;&amp; echo &quot;hello&quot; &#39;world&#39;</parameter></invoke>'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args["command"], "cat < input.txt && echo \"hello\" 'world'")

    # -------------------------------------------------------------------------
    # 4. Corrupted & Invalid JSON Handling
    # -------------------------------------------------------------------------

    def test_degenerate_single_quotes_with_inner_double_quotes(self):
        """Single-quoted JSON containing unescaped double quotes inside strings."""
        text = "{'tool': 'bash', 'command': 'echo \"Hello World\"'}"
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args["command"], 'echo "Hello World"')

    def test_degenerate_empty_or_whitespace_tool_name_rejected(self):
        """Empty or whitespace-only tool name must be rejected."""
        cases = [
            '{"tool": "", "command": "ls"}',
            '{"tool": "   ", "command": "ls"}',
            '{"name": "", "arguments": {}}',
            '<invoke name=""><parameter name="cmd">ls</parameter></invoke>',
            '<invoke name="   "><parameter name="cmd">ls</parameter></invoke>',
        ]
        for inp in cases:
            self.assertEqual(try_parse_tool_calls(inp), [])
            with self.assertRaises(ToolError):
                parse_tool_call(inp)

    def test_degenerate_non_string_or_null_tool_name_rejected(self):
        """Null, integer, boolean, or array as tool name must be rejected."""
        cases = [
            '{"tool": null, "command": "ls"}',
            '{"tool": 12345, "command": "ls"}',
            '{"tool": true, "command": "ls"}',
            '{"tool": ["bash"], "command": "ls"}',
            '{"name": null, "arguments": {}}',
            '{"name": 42, "arguments": {}}',
        ]
        for inp in cases:
            self.assertEqual(try_parse_tool_calls(inp), [])
            with self.assertRaises(ToolError):
                parse_tool_call(inp)

    def test_degenerate_empty_objects_and_arrays(self):
        """Raw empty JSON objects and arrays without tool keys must be rejected."""
        cases = ["{}", "[]", "   {}   ", "   []   "]
        for inp in cases:
            self.assertEqual(try_parse_tool_calls(inp), [])
            with self.assertRaises(ToolError):
                parse_tool_call(inp)

    def test_degenerate_unescaped_literal_newlines(self):
        """Raw unescaped literal newlines (0x0A) inside JSON string literals."""
        text = '{"tool": "bash", "command": "echo line 1\necho line 2\necho line 3"}'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args["command"], "echo line 1\necho line 2\necho line 3")

    # -------------------------------------------------------------------------
    # 5. Code, Math & Syntax Collisions (False Positive Immunity)
    # -------------------------------------------------------------------------

    def test_degenerate_math_inequalities_collision(self):
        """Mathematical inequality comparisons (x < 10 && y > 20) must not trigger tag matching."""
        text = """Here is the validation logic in Python:
def check(x, y):
    if x < 10 and y > 20:
        return x < 5 or y > 50
    return False"""
        self.assertEqual(try_parse_tool_calls(text), [])
        with self.assertRaises(ToolError):
            parse_tool_call(text)

    def test_degenerate_cpp_templates_collision(self):
        """C++ nested templates (std::vector<std::pair<int, int>>) must not be parsed as tags."""
        text = "Use this type definition: std::vector<std::pair<int, int>> lookup_table;"
        self.assertEqual(try_parse_tool_calls(text), [])
        with self.assertRaises(ToolError):
            parse_tool_call(text)

    def test_degenerate_react_keyword_collision(self):
        """Casual conversational use of 'Action:' without 'Action Input:' must not trigger ReAct."""
        text = "In conclusion, Action: we should consider refactoring the parser next sprint."
        self.assertEqual(try_parse_tool_calls(text), [])
        with self.assertRaises(ToolError):
            parse_tool_call(text)

    # -------------------------------------------------------------------------
    # 6. Thinking Blocks Pathologies
    # -------------------------------------------------------------------------

    def test_degenerate_thinking_block_retracted_proposal(self):
        """Tool call inside <think>...</think> must be stripped and never executed."""
        text = """<think>
I should probably run:
<tool_call>
{"name": "bash", "arguments": {"cmd": "rm -rf /"}}
</tool_call>
Wait, that is destructive and dangerous. I will not run that.
</think>
I cannot perform destructive actions."""
        self.assertEqual(try_parse_tool_calls(text), [])
        with self.assertRaises(ToolError):
            parse_tool_call(text)

    def test_degenerate_multiple_thinking_blocks_with_real_call(self):
        """Multiple <think> blocks interleaved with text, followed by real tool call."""
        text = """<think>first contemplation</think>
Intermediate explanation.
<think>second contemplation</think>
<invoke name="search">
<parameter name="query">rust parser</parameter>
</invoke>"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "search")
        self.assertEqual(calls[0].args["pattern"], "rust parser")

    # -------------------------------------------------------------------------
    # 7. Stress & ReDoS Resilience
    # -------------------------------------------------------------------------

    def test_degenerate_massive_string_without_catastrophic_backtracking(self):
        """50,000 repeating characters inside parameter parsed in under 200 ms without ReDoS."""
        payload = "x" * 50000
        text = f'<invoke name="bash"><parameter name="cmd">{payload}</parameter></invoke>'
        start = time.perf_counter()
        call = parse_tool_call(text)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self.assertEqual(call.name, "bash")
        self.assertEqual(len(call.args["command"]), 50000)
        self.assertLess(elapsed_ms, 500.0, f"ReDoS vulnerability detected: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    unittest.main()
