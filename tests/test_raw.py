# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

import json
import unittest

from agent_tool_parser import (
    RawToolCall,
    ToolError,
    extract_raw_tool_calls,
    try_extract_raw_tool_call,
)


class TestRawToolCall(unittest.TestCase):
    def test_raw_tool_call_basic(self):
        raw = RawToolCall(
            name="read_file",
            raw_args='{"path": "src/main.py"}',
            raw_source='<invoke name="read_file">{"path": "src/main.py"}</invoke>',
            call_id="call_123",
        )
        self.assertEqual(raw.name, "read_file")
        self.assertEqual(raw.raw_args, '{"path": "src/main.py"}')
        self.assertEqual(raw.call_id, "call_123")
        self.assertEqual(raw.parse_args(), {"path": "src/main.py"})

        tc = raw.to_tool_call()
        self.assertEqual(tc.name, "read_file")
        self.assertEqual(tc.args, {"path": "src/main.py"})
        self.assertEqual(tc.raw_source, raw.raw_source)

    def test_raw_tool_call_invalid_json_args(self):
        raw = RawToolCall(
            name="broken",
            raw_args='{"broken": ',
            raw_source="broken",
        )
        with self.assertRaises(ToolError):
            raw.parse_args()

        with self.assertRaises(ToolError):
            raw.to_tool_call()

    def test_extract_raw_dsml(self):
        text = """<｜DSML｜tool_calls>
<｜DSML｜tool name=search>
<｜DSML｜parameter name=pattern string=true>ignore-paths</｜DSML｜parameter>
<｜DSML｜parameter name=path string=true>.</｜DSML｜parameter>
</｜DSML｜tool>
</｜DSML｜tool_calls>"""
        raw_calls = extract_raw_tool_calls(text)
        self.assertEqual(len(raw_calls), 1)
        self.assertEqual(raw_calls[0].name, "search")
        parsed = raw_calls[0].parse_args()
        self.assertEqual(parsed, {"pattern": "ignore-paths", "path": "."})

        single = try_extract_raw_tool_call(text)
        self.assertIsNotNone(single)
        self.assertEqual(single.name, "search")

    def test_extract_raw_markdown_json(self):
        payload = {
            "name": "bash",
            "arguments": {"command": "cargo test"},
        }
        text = f"Here is the command:\n```json\n{json.dumps(payload)}\n```\nDone."
        raw_calls = extract_raw_tool_calls(text)
        self.assertEqual(len(raw_calls), 1)
        self.assertEqual(raw_calls[0].name, "bash")
        self.assertEqual(raw_calls[0].parse_args(), {"command": "cargo test"})

    def test_extract_raw_tool_call_xml(self):
        text = (
            '<tool_call>\n{"name": "fetch_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        )
        raw_calls = extract_raw_tool_calls(text)
        self.assertEqual(len(raw_calls), 1)
        self.assertEqual(raw_calls[0].name, "fetch_weather")
        self.assertEqual(raw_calls[0].parse_args(), {"city": "Paris"})

    def test_extract_raw_empty_when_no_tool(self):
        text = "Hello! I am a helpful AI assistant. How can I help you today?"
        raw_calls = extract_raw_tool_calls(text)
        self.assertEqual(len(raw_calls), 0)
        self.assertIsNone(try_extract_raw_tool_call(text))

    def test_extract_raw_multiple_direct_json(self):
        text = (
            "I will run two commands:\n"
            '{"name": "read_file", "arguments": {"path": "main.rs"}}\n'
            '{"name": "bash", "arguments": {"command": "cargo test"}}'
        )
        calls = extract_raw_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[1].name, "bash")


if __name__ == "__main__":
    unittest.main()
