"""Comprehensive Functional, Integration, and QA Test Suite for agent-tool-parser.

Validates end-to-end multi-agent pipelines, cross-provider translation workflows,
MCP JSON-RPC bridges, A2A governance headers, and boundary/adversarial edge cases.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from agent_tool_parser import (
    ProtocolRegistry,
    ToolCategory,
    ToolMetadata,
    canonicalize_tool_calls,
    convert_tool_calls,
    convert_tool_results,
    create_streaming_transformer,
    sanitize_tool_schema_for_gemini,
    to_mcp_call_result,
    to_mcp_tool_call,
    translate_request,
    try_parse_tool_calls,
)


class TestFunctionalEndToEndGateway(unittest.TestCase):
    """End-to-end integration tests simulating a production AI gateway / reverse proxy."""

    def test_e2e_openai_to_gemini_request_and_streaming_pipeline(self):
        """Simulate incoming OpenAI ChatCompletion request -> Gemini translation -> streaming response back."""
        # 1. Incoming OpenAI request from client with tools
        openai_req: dict[str, Any] = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are an autonomous engineering assistant."},
                {"role": "user", "content": "Run test suite and check code quality."},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "description": "Executes a shell command in sandbox",
                        "parameters": {
                            "$schema": "https://json-schema.org/draft/2020-12/schema",
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "Shell command to execute",
                                },
                                "timeout": {
                                    "anyOf": [
                                        {"type": "integer", "description": "Timeout in seconds"},
                                        {"type": "null"},
                                    ]
                                },
                            },
                            "required": ["command"],
                            "additionalProperties": False,
                            "$defs": {
                                "UnusedRef": {"type": "string"},
                            },
                        },
                    },
                }
            ],
        }

        # 2. Gateway translates request to Gemini
        gemini_req = translate_request(
            openai_req,
            from_format="openai",
            to_format="gemini",
            override_model="gemini-2.5-pro",
        )

        # Verify Gemini structure
        self.assertIn("contents", gemini_req)
        self.assertIn("system_instruction", gemini_req)
        self.assertEqual(gemini_req["contents"][0]["role"], "user")
        self.assertIn("tools", gemini_req)

        # Verify Gemini schema was automatically sanitized
        gemini_fn = gemini_req["tools"][0]["function_declarations"][0]
        self.assertEqual(gemini_fn["name"], "run_command")
        params = gemini_fn["parameters"]
        self.assertNotIn("$schema", params)
        self.assertNotIn("$defs", params)
        self.assertNotIn("additionalProperties", params)
        self.assertNotIn("anyOf", params["properties"]["timeout"])
        self.assertEqual(params["properties"]["timeout"]["type"], "integer")

        # 3. Simulate streaming response from Gemini upstream
        transformer = create_streaming_transformer("gemini", "openai", model="gpt-4o")

        raw_gemini_sse_chunks = [
            # Chunk 1: Reasoning text
            'data: {"candidates": [{"content": {"parts": [{"text": "I will execute pytest first."}]}}]}\n\n',
            # Chunk 2: Function call invocation
            'data: {"candidates": [{"content": {"parts": [{"functionCall": {"name": "run_command", "args": {"command": "pytest -v"}}}]}}]}\n\n',
            # Chunk 3: Finish reason
            'data: {"candidates": [{"finishReason": "STOP"}]}\n\n',
        ]

        received_openai_deltas: list[dict[str, Any]] = []
        for line in raw_gemini_sse_chunks:
            out_lines = transformer.transform_sse_line(line)
            for out_line in out_lines:
                if out_line.startswith("data: ") and not out_line.startswith("data: [DONE]"):
                    payload = json.loads(out_line[6:].strip())
                    choice = payload["choices"][0]
                    received_openai_deltas.append(choice)

        self.assertGreaterEqual(len(received_openai_deltas), 2)
        # Verify text chunk received
        text_deltas = [
            d["delta"].get("content") for d in received_openai_deltas if d["delta"].get("content")
        ]
        self.assertEqual(text_deltas, ["I will execute pytest first."])

        # Verify tool_call chunk received
        tool_deltas = [
            d["delta"].get("tool_calls")
            for d in received_openai_deltas
            if d["delta"].get("tool_calls")
        ]
        self.assertTrue(len(tool_deltas) > 0)
        tc = tool_deltas[0][0]
        self.assertEqual(tc["function"]["name"], "run_command")
        self.assertIn('"command"', tc["function"]["arguments"])

    def test_e2e_multi_turn_history_with_tool_results_round_trip(self):
        """Simulate a full 4-turn conversation history with tool execution results."""
        # Multi-turn conversation in OpenAI format
        openai_history: dict[str, Any] = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What is the disk usage in /var/log?"},
                {
                    "role": "assistant",
                    "content": "Checking disk space now.",
                    "tool_calls": [
                        {
                            "id": "call_disk_01",
                            "type": "function",
                            "function": {"name": "check_disk", "arguments": '{"path": "/var/log"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_disk_01",
                    "name": "check_disk",
                    "content": '{"used_mb": 420, "free_mb": 8500, "status": "ok"}',
                },
                {
                    "role": "assistant",
                    "content": "The disk usage in /var/log is 420 MB with 8.5 GB free.",
                },
            ],
        }

        # 1. Translate OpenAI -> Anthropic
        anthropic_req = translate_request(
            openai_history, from_format="openai", to_format="anthropic"
        )
        self.assertEqual(len(anthropic_req["messages"]), 4)
        # Turn 0: user
        self.assertEqual(anthropic_req["messages"][0]["role"], "user")
        # Turn 1: assistant with tool_use block
        self.assertEqual(anthropic_req["messages"][1]["role"], "assistant")
        self.assertTrue(
            any(b.get("type") == "tool_use" for b in anthropic_req["messages"][1]["content"])
        )
        # Turn 2: user with tool_result block
        self.assertEqual(anthropic_req["messages"][2]["role"], "user")
        self.assertTrue(
            any(b.get("type") == "tool_result" for b in anthropic_req["messages"][2]["content"])
        )
        # Turn 3: assistant with final text
        self.assertEqual(anthropic_req["messages"][3]["role"], "assistant")

        # 2. Translate Anthropic -> Gemini
        gemini_req = translate_request(anthropic_req, from_format="anthropic", to_format="gemini")
        self.assertEqual(len(gemini_req["contents"]), 4)
        # Turn 1: model with functionCall
        self.assertEqual(gemini_req["contents"][1]["role"], "model")
        self.assertTrue(any("functionCall" in p for p in gemini_req["contents"][1]["parts"]))
        # Turn 2: user with functionResponse resolved to tool name 'check_disk'
        self.assertEqual(gemini_req["contents"][2]["role"], "user")
        fr_part = next(
            p["functionResponse"]
            for p in gemini_req["contents"][2]["parts"]
            if "functionResponse" in p
        )
        self.assertEqual(fr_part["name"], "check_disk")

        # 3. Translate Gemini -> OpenAI (Full Round-Trip)
        openai_restored = translate_request(gemini_req, from_format="gemini", to_format="openai")
        self.assertEqual(len(openai_restored["messages"]), 4)
        self.assertEqual(openai_restored["messages"][0]["role"], "user")
        self.assertEqual(openai_restored["messages"][1]["role"], "assistant")
        self.assertTrue(len(openai_restored["messages"][1].get("tool_calls", [])) > 0)
        self.assertEqual(openai_restored["messages"][2]["role"], "tool")


class TestMCPAndA2AGovernanceQA(unittest.TestCase):
    """Rigorous QA tests for MCP JSON-RPC bridges and Agent-to-Agent governance."""

    def test_mcp_heterogeneous_tool_call_extraction_and_result_cycle(self):
        """Extract tool calls from different model formats into MCP JSON-RPC and process results."""
        # DeepSeek DSML format
        deepseek_output = (
            "<think>Need to search repo symbols.</think>\n"
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="search_code">\n'
            '<｜DSML｜parameter name="symbol">ToolMetadata</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="max_results">5</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )

        # 1. to_mcp_tool_call directly from raw text
        mcp_call = to_mcp_tool_call(deepseek_output)
        self.assertEqual(mcp_call["jsonrpc"], "2.0")
        self.assertEqual(mcp_call["method"], "tools/call")
        self.assertEqual(mcp_call["params"]["name"], "search_code")
        self.assertEqual(mcp_call["params"]["arguments"]["symbol"], "ToolMetadata")
        self.assertEqual(int(mcp_call["params"]["arguments"]["max_results"]), 5)

        # 2. Simulate tool execution and create MCP CallToolResult
        mock_tool_output = [{"symbol": "ToolMetadata", "file": "universal.py", "line": 1640}]
        mcp_result = to_mcp_call_result(mock_tool_output, is_error=False)
        self.assertFalse(mcp_result["isError"])
        self.assertEqual(mcp_result["content"][0]["type"], "text")
        self.assertIn("universal.py", mcp_result["content"][0]["text"])

        # 3. Convert MCP result into Anthropic tool_result
        anth_results = convert_tool_results(
            mcp_result,
            from_format="mcp",
            to_format="anthropic",
            default_tool_call_id="call_mcp_99",
        )
        self.assertEqual(len(anth_results), 1)
        self.assertEqual(anth_results[0]["type"], "tool_result")
        self.assertEqual(anth_results[0]["tool_use_id"], "call_mcp_99")
        self.assertFalse(anth_results[0]["is_error"])

        # 4. Error result handling
        err_res = to_mcp_call_result("File not found on disk", is_error=True)
        self.assertTrue(err_res["isError"])
        openai_err_msgs = convert_tool_results(err_res, from_format="mcp", to_format="openai")
        self.assertEqual(len(openai_err_msgs), 1)
        self.assertEqual(openai_err_msgs[0]["role"], "tool")
        self.assertIn("File not found on disk", openai_err_msgs[0]["content"])

    def test_a2a_governance_headers_and_security_policy_enforcement(self):
        """Simulate a multi-agent delegation pipeline with security policy verification."""
        # Agent A (Planner) generates a mutating task definition
        planner_meta = ToolMetadata(
            caller_agent="agent-planner-01",
            target_agent="agent-worker-02",
            task_id="task-deploy-789",
            parent_call_id="call-root-100",
            category=ToolCategory.EDIT,
            is_mutation=True,
            is_verification=False,
            requires_approval=True,
            idempotent=False,
            security_level="restricted",
            artifact_ids=["art-patch-01", "art-spec-02"],
            custom_annotations={
                "cluster": "prod-useast",
                "compliance-check": "soc2",
            },
        )

        # Serialize to HTTP / RPC metadata headers
        headers = planner_meta.to_headers(prefix="x-a2a-")
        self.assertEqual(headers["x-a2a-caller-agent"], "agent-planner-01")
        self.assertEqual(headers["x-a2a-target-agent"], "agent-worker-02")
        self.assertEqual(headers["x-a2a-task-id"], "task-deploy-789")
        self.assertEqual(headers["x-a2a-requires-approval"], "true")
        self.assertEqual(headers["x-a2a-is-mutation"], "true")
        self.assertEqual(headers["x-a2a-security-level"], "restricted")
        self.assertEqual(headers["x-a2a-artifact-ids"], "art-patch-01,art-spec-02")
        self.assertEqual(headers["x-a2a-meta-compliance-check"], "soc2")

        # Receiver parses headers
        received_meta = ToolMetadata.from_headers(headers, prefix="x-a2a-")
        self.assertEqual(received_meta.caller_agent, "agent-planner-01")
        self.assertEqual(received_meta.target_agent, "agent-worker-02")
        self.assertTrue(received_meta.requires_approval)
        self.assertTrue(received_meta.is_mutation)
        self.assertFalse(received_meta.is_verification)
        self.assertEqual(received_meta.security_level, "restricted")
        self.assertEqual(received_meta.artifact_ids, ["art-patch-01", "art-spec-02"])
        self.assertEqual(received_meta.custom_annotations["compliance-check"], "soc2")

        # Security policy gate verification
        def execute_tool_policy_gate(meta: ToolMetadata) -> str:
            if meta.security_level == "restricted" and meta.requires_approval:
                return "BLOCKED_PENDING_HUMAN_APPROVAL"
            return "EXECUTED"

        self.assertEqual(execute_tool_policy_gate(received_meta), "BLOCKED_PENDING_HUMAN_APPROVAL")


class TestBoundaryAdversarialAndEdgeCases(unittest.TestCase):
    """Boundary, degenerate, and adversarial input testing."""

    def test_schema_sanitizer_edge_cases(self):
        """Test sanitizer on degenerate, empty, and deeply nested schemas."""
        # 1. Empty schema
        self.assertEqual(sanitize_tool_schema_for_gemini({}), {})

        # 2. Schema with missing or broken $ref (should not crash)
        broken_ref_schema = {
            "type": "object",
            "properties": {
                "broken": {"$ref": "#/$defs/DoesNotExist"},
            },
        }
        sanitized = sanitize_tool_schema_for_gemini(broken_ref_schema)
        self.assertIn("broken", sanitized["properties"])

        # 3. Schema with non-dict properties
        weird_schema = {
            "type": "object",
            "properties": {
                "primitive_val": "string",  # invalid schema type
            },
        }
        res = sanitize_tool_schema_for_gemini(weird_schema)
        self.assertIn("primitive_val", res["properties"])

        # 4. Deeply nested anyOf flattening
        nested_anyof = {
            "type": "object",
            "properties": {
                "field": {
                    "anyOf": [
                        {"type": "null"},
                        {
                            "anyOf": [
                                {"type": "string", "description": "deep string"},
                                {"type": "null"},
                            ]
                        },
                    ]
                }
            },
        }
        sanitized_nested = sanitize_tool_schema_for_gemini(nested_anyof)
        field_prop = sanitized_nested["properties"]["field"]
        self.assertNotIn("anyOf", field_prop)
        self.assertEqual(field_prop["type"], "string")

    def test_unicode_multilingual_and_emoji_arguments(self):
        """Ensure multi-byte unicode, emojis, and international text are parsed without corruption."""
        raw_unicode = """
        <tool_call>
        {"name": "translate_text", "arguments": {"source": "Hello world! 🚀 Bonjour le monde, こんにちは世界", "target_lang": "中文", "count": 42}}
        </tool_call>
        """
        calls = try_parse_tool_calls(raw_unicode)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "translate_text")
        self.assertEqual(
            calls[0].args["source"], "Hello world! 🚀 Bonjour le monde, こんにちは世界"
        )
        self.assertEqual(calls[0].args["target_lang"], "中文")
        self.assertEqual(calls[0].args["count"], 42)

    def test_unescaped_shell_quotes_and_code_blocks(self):
        """Repair inner unescaped double quotes inside shell and Python script arguments."""
        corrupt_raw = (
            "<tool_call>\n"
            '{"name": "bash", "arguments": {"cmd": "echo "nested unescaped double quotes" && ls"}}\n'
            "</tool_call>"
        )
        calls = try_parse_tool_calls(corrupt_raw)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "bash")
        cmd_val = calls[0].args.get("command") or calls[0].args.get("cmd")
        self.assertIn("nested unescaped double quotes", cmd_val)

    def test_kv_cache_canonicalization_stability(self):
        """Verify that differently ordered dictionary arguments produce identical sorted keys."""
        calls1 = [
            {"type": "function", "function": {"name": "z_tool", "arguments": '{"z": 1, "a": 2}'}},
            {"type": "function", "function": {"name": "a_tool", "arguments": '{"x": 10, "m": 5}'}},
        ]
        calls2 = [
            {"type": "function", "function": {"name": "a_tool", "arguments": '{"m": 5, "x": 10}'}},
            {"type": "function", "function": {"name": "z_tool", "arguments": '{"a": 2, "z": 1}'}},
        ]

        c1 = canonicalize_tool_calls(calls1)
        c2 = canonicalize_tool_calls(calls2)

        # Deterministic order by tool name, and deterministic sorted argument keys
        self.assertEqual(c1, c2)
        self.assertEqual(c1[0]["function"]["name"], "a_tool")
        self.assertEqual(c1[0]["function"]["arguments"], '{"m":5,"x":10}')
        self.assertEqual(c1[1]["function"]["name"], "z_tool")
        self.assertEqual(c1[1]["function"]["arguments"], '{"a":2,"z":1}')

    def test_provider_alias_registry_custom_registration_and_reset(self):
        """Verify adding custom internal provider aliases and isolation."""
        ProtocolRegistry.register_alias("inboost_llm_worker", "openai")
        self.assertEqual(ProtocolRegistry.get("inboost_llm_worker").protocol_name, "openai")

        ProtocolRegistry.reset_aliases()
        with self.assertRaises(ValueError) as ctx:
            ProtocolRegistry.get("inboost_llm_worker")
        self.assertIn("Unknown protocol or provider: 'inboost_llm_worker'", str(ctx.exception))


class TestQwenInClaudeCodeWorkflow(unittest.TestCase):
    """Tests simulating Qwen 2.5 open-weights model deployed inside Claude Code (Anthropic protocol)."""

    def test_claude_code_request_translation_to_qwen(self):
        """Translate Claude Code's Anthropic Messages request into Qwen OpenAI-compatible request."""
        claude_code_req = {
            "model": "claude-3-5-sonnet",
            "system": "You are Claude Code, an agentic coding assistant.",
            "messages": [{"role": "user", "content": "Update the version in setup.py to 0.2.0"}],
            "tools": [
                {
                    "name": "Edit",
                    "description": "Performs string replacement on files",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["file_path", "old_string", "new_string"],
                    },
                },
                {
                    "name": "Bash",
                    "description": "Executes shell commands",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                },
            ],
        }

        # Route to Qwen via OpenAI-compatible endpoint
        qwen_req = translate_request(
            claude_code_req,
            from_format="anthropic",
            to_format="openai",
            override_model="qwen-2.5-coder-32b",
        )

        self.assertEqual(qwen_req["model"], "qwen-2.5-coder-32b")
        self.assertEqual(len(qwen_req["messages"]), 2)  # system + user
        self.assertEqual(qwen_req["messages"][0]["role"], "system")
        self.assertEqual(qwen_req["messages"][1]["role"], "user")
        self.assertEqual(len(qwen_req["tools"]), 2)
        self.assertEqual(qwen_req["tools"][0]["function"]["name"], "Edit")
        self.assertEqual(qwen_req["tools"][1]["function"]["name"], "Bash")

    def test_qwen_xml_output_to_claude_code_tool_use(self):
        """When Qwen outputs XML syntax, convert directly to Anthropic tool_use for Claude Code."""
        qwen_xml_output = (
            "I will update setup.py.\n"
            "<function=Edit>\n"
            "<parameter=file_path>setup.py</parameter>\n"
            '<parameter=old_string>version="0.1.6"</parameter>\n'
            '<parameter=new_string>version="0.2.0"</parameter>\n'
            "</function>"
        )

        anthropic_calls = convert_tool_calls(
            qwen_xml_output, from_format="raw", to_format="anthropic"
        )
        self.assertEqual(len(anthropic_calls), 1)
        self.assertEqual(anthropic_calls[0]["type"], "tool_use")
        # Canonical normalization resolves 'Edit' to 'str_replace' and 'file_path' to 'path'
        self.assertEqual(anthropic_calls[0]["name"], "str_replace")
        self.assertEqual(anthropic_calls[0]["input"]["path"], "setup.py")
        self.assertEqual(anthropic_calls[0]["input"]["old_string"], 'version="0.1.6"')
        self.assertEqual(anthropic_calls[0]["input"]["new_string"], 'version="0.2.0"')

    def test_qwen_action_tokens_to_claude_code_tool_use(self):
        """When Qwen outputs <|action_start|> tokens, convert directly to Anthropic tool_use."""
        qwen_action_output = (
            "<|action_start|><|action_name|>Bash<|action_args|>"
            '{"command": "pytest tests/test_parser.py"}'
            "<|action_end|>"
        )

        anthropic_calls = convert_tool_calls(
            qwen_action_output, from_format="raw", to_format="anthropic"
        )
        self.assertEqual(len(anthropic_calls), 1)
        self.assertEqual(anthropic_calls[0]["type"], "tool_use")
        # Canonical normalization resolves 'Bash' to lowercase 'bash'
        self.assertEqual(anthropic_calls[0]["name"], "bash")
        self.assertEqual(anthropic_calls[0]["input"]["command"], "pytest tests/test_parser.py")

    def test_qwen_container_tool_calls_to_claude_code_tool_use(self):
        """When Qwen outputs <tool_call><function=...>, convert cleanly to Anthropic tool_use."""
        qwen_container_output = (
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>git diff</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

        anthropic_calls = convert_tool_calls(
            qwen_container_output, from_format="raw", to_format="anthropic"
        )
        self.assertEqual(len(anthropic_calls), 1)
        self.assertEqual(anthropic_calls[0]["type"], "tool_use")
        self.assertEqual(anthropic_calls[0]["name"], "bash")
        self.assertEqual(anthropic_calls[0]["input"]["command"], "git diff")

    def test_claude_code_tool_result_conversion_to_qwen(self):
        """When Claude Code executes tool and replies with Anthropic tool_result, convert back to OpenAI for Qwen."""
        claude_tool_result = {
            "type": "tool_result",
            "tool_use_id": "call_edit_01",
            "content": "File setup.py replaced successfully: 1 match.",
            "is_error": False,
        }

        openai_tool_msgs = convert_tool_results(
            claude_tool_result,
            from_format="anthropic",
            to_format="openai",
            tool_name_map={"call_edit_01": "Edit"},
        )
        self.assertEqual(len(openai_tool_msgs), 1)
        self.assertEqual(openai_tool_msgs[0]["role"], "tool")
        self.assertEqual(openai_tool_msgs[0]["tool_call_id"], "call_edit_01")
        self.assertEqual(openai_tool_msgs[0]["name"], "Edit")
        self.assertIn("File setup.py replaced successfully", openai_tool_msgs[0]["content"])


if __name__ == "__main__":
    unittest.main()
