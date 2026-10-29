# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

import json
import unittest

from agent_tool_parser import (
    ToolCall,
    canonical_json_dumps,
    canonical_sort_keys,
    canonicalize_arguments_string,
    canonicalize_tool_calls,
)


class TestCanonical(unittest.TestCase):
    def test_canonical_sort_keys_nested(self):
        data = {
            "z": 100,
            "m": {"b": 2, "a": 1},
            "items": [
                {"y": 2, "x": 1},
                {"k": [3, 2, 1]},
            ],
        }
        sorted_data = canonical_sort_keys(data)
        self.assertEqual(list(sorted_data.keys()), ["items", "m", "z"])
        self.assertEqual(list(sorted_data["m"].keys()), ["a", "b"])
        self.assertEqual(list(sorted_data["items"][0].keys()), ["x", "y"])

    def test_canonical_json_dumps_determinism(self):
        # Two dictionaries with identical key-values but different insertion orders
        d1 = {"path": "src/main.py", "action": "write", "content": "print(1)"}
        d2 = {"content": "print(1)", "action": "write", "path": "src/main.py"}
        d3 = {"action": "write", "path": "src/main.py", "content": "print(1)"}

        s1 = canonical_json_dumps(d1)
        s2 = canonical_json_dumps(d2)
        s3 = canonical_json_dumps(d3)

        self.assertEqual(s1, s2)
        self.assertEqual(s2, s3)
        self.assertEqual(s1, '{"action":"write","content":"print(1)","path":"src/main.py"}')

    def test_canonicalize_arguments_string(self):
        raw_str = '{\n  "timeout": 30,\n  "command": "git status"\n}'
        canon = canonicalize_arguments_string(raw_str)
        self.assertEqual(canon, '{"command":"git status","timeout":30}')

        # Non-json strings returned stripped
        self.assertEqual(canonicalize_arguments_string("not json"), "not json")
        self.assertEqual(canonicalize_arguments_string(""), "")

    def test_tool_call_to_canonical_json(self):
        tc = ToolCall(name="edit_file", args={"target": "app.py", "old": "foo", "new": "bar"})
        self.assertEqual(tc.to_canonical_json(), '{"new":"bar","old":"foo","target":"app.py"}')

    def test_tool_call_to_openai_tool_call(self):
        tc = ToolCall(name="bash", args={"command": "cargo build", "timeout": 60})
        openai_format = tc.to_openai_tool_call(call_id="call_99", canonical=True)

        self.assertEqual(
            openai_format,
            {
                "id": "call_99",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command":"cargo build","timeout":60}',
                },
            },
        )

        # Default ID generation
        default_format = tc.to_openai_tool_call()
        self.assertEqual(default_format["id"], "call_bash")

    def test_canonicalize_tool_calls_stable_sorting(self):
        calls = [
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "a.txt", "content": "hello"}',
                },
            },
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"offset": 10, "path": "b.txt"}',
                },
            },
        ]

        canonical = canonicalize_tool_calls(calls)
        self.assertEqual(len(canonical), 2)
        # Should be sorted by function name: read_file before write_file
        self.assertEqual(canonical[0]["function"]["name"], "read_file")
        self.assertEqual(canonical[0]["function"]["arguments"], '{"offset":10,"path":"b.txt"}')
        self.assertEqual(canonical[1]["function"]["name"], "write_file")
        self.assertEqual(
            canonical[1]["function"]["arguments"], '{"content":"hello","path":"a.txt"}'
        )

    def test_kv_cache_prefix_invariance(self):
        """Verify that turns generated with permuted argument orders produce identical serialized bytes."""
        payload_a = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "edit", "arguments": '{"b": 2, "a": 1}'},
                    }
                ],
            }
        ]
        payload_b = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "edit", "arguments": '{"a": 1, "b": 2}'},
                    }
                ],
            }
        ]

        norm_a = canonicalize_tool_calls(payload_a[0]["tool_calls"])
        norm_b = canonicalize_tool_calls(payload_b[0]["tool_calls"])

        # Byte-level equivalence
        bytes_a = json.dumps(norm_a, sort_keys=True).encode("utf-8")
        bytes_b = json.dumps(norm_b, sort_keys=True).encode("utf-8")

        self.assertEqual(bytes_a, bytes_b)
