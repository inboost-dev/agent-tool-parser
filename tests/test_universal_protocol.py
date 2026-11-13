"""Comprehensive test suite for Universal Protocol IR, Adapters, and Streaming Transformers."""

from __future__ import annotations

import unittest

from agent_tool_parser import (
    BaseProtocolAdapter,
    CanonicalChunk,
    CanonicalRequest,
    CanonicalResponse,
    ProtocolRegistry,
    ToolDefinition,
    ToolMetadata,
    UniversalProtocolAdapter,
    convert_tool_calls,
    convert_tool_definitions,
    convert_tool_results,
    create_streaming_transformer,
    normalize_to_ir,
    sanitize_tool_schema_for_gemini,
    sanitize_tool_schema_for_openai,
    to_mcp_call_result,
    to_mcp_tool_call,
    translate_request,
    translate_response,
)


class TestUniversalProtocolIR(unittest.TestCase):
    """Tests for Request and Response translation across OpenAI, Anthropic, and Gemini."""

    def test_openai_to_anthropic_request(self):
        openai_payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a code refactoring assistant."},
                {"role": "user", "content": "Refactor auth_handler"},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "description": "Edits file content",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ],
        }

        anthropic_req = translate_request(
            openai_payload,
            from_format="openai",
            to_format="anthropic",
            override_model="claude-3-5-sonnet",
        )
        self.assertEqual(anthropic_req["model"], "claude-3-5-sonnet")
        self.assertEqual(anthropic_req["system"], "You are a code refactoring assistant.")
        self.assertEqual(len(anthropic_req["messages"]), 1)
        self.assertEqual(anthropic_req["messages"][0]["role"], "user")
        self.assertEqual(anthropic_req["messages"][0]["content"], "Refactor auth_handler")
        self.assertEqual(len(anthropic_req["tools"]), 1)
        self.assertEqual(anthropic_req["tools"][0]["name"], "edit_file")
        self.assertIn("input_schema", anthropic_req["tools"][0])

    def test_anthropic_to_gemini_request_and_name_resolution(self):
        anthropic_payload = {
            "model": "claude-3-5-sonnet",
            "system": "System instructions here.",
            "messages": [
                {"role": "user", "content": "Hello Gemini"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Calling tool now."},
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "bash",
                            "input": {"cmd": "ls -l"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": "file1.txt\nfile2.txt",
                        },
                    ],
                },
            ],
            "temperature": 0.5,
        }

        gemini_req = translate_request(
            anthropic_payload,
            from_format="anthropic",
            to_format="gemini",
            override_model="gemini-2.5-pro",
        )
        self.assertIn("system_instruction", gemini_req)
        self.assertEqual(
            gemini_req["system_instruction"]["parts"][0]["text"], "System instructions here."
        )
        self.assertEqual(len(gemini_req["contents"]), 3)

        # Turn 0: user
        self.assertEqual(gemini_req["contents"][0]["role"], "user")
        # Turn 1: model with functionCall
        self.assertEqual(gemini_req["contents"][1]["role"], "model")
        fc = [p["functionCall"] for p in gemini_req["contents"][1]["parts"] if "functionCall" in p]
        self.assertEqual(len(fc), 1)
        self.assertEqual(fc[0]["name"], "bash")
        self.assertEqual(fc[0]["args"], {"cmd": "ls -l"})
        # Turn 2: user with functionResponse resolved to tool name 'bash'
        self.assertEqual(gemini_req["contents"][2]["role"], "user")
        fr = [
            p["functionResponse"]
            for p in gemini_req["contents"][2]["parts"]
            if "functionResponse" in p
        ]
        self.assertEqual(len(fr), 1)
        self.assertEqual(fr[0]["name"], "bash")
        self.assertEqual(fr[0]["response"], {"content": "file1.txt\nfile2.txt"})

    def test_gemini_to_openai_request(self):
        gemini_payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Execute search"}]},
                {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "grep", "args": {"pattern": "def "}}}],
                },
            ]
        }

        openai_req = translate_request(
            gemini_payload, from_format="gemini", to_format="openai", override_model="gpt-4o"
        )
        self.assertEqual(openai_req["model"], "gpt-4o")
        self.assertEqual(len(openai_req["messages"]), 2)
        self.assertEqual(openai_req["messages"][0]["role"], "user")
        self.assertEqual(openai_req["messages"][1]["role"], "assistant")
        self.assertIn("tool_calls", openai_req["messages"][1])
        self.assertEqual(openai_req["messages"][1]["tool_calls"][0]["function"]["name"], "grep")

    def test_multimodal_request_translation(self):
        openai_multimodal = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                            },
                        },
                    ],
                }
            ],
        }

        # OpenAI -> Gemini
        gemini_req = translate_request(openai_multimodal, from_format="openai", to_format="gemini")
        parts = gemini_req["contents"][0]["parts"]
        self.assertEqual(parts[0]["text"], "Inspect this image")
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "image/png")
        self.assertTrue(parts[1]["inline_data"]["data"].startswith("iVBORw"))

        # OpenAI -> Anthropic
        anthropic_req = translate_request(
            openai_multimodal, from_format="openai", to_format="anthropic"
        )
        blocks = anthropic_req["messages"][0]["content"]
        self.assertEqual(blocks[0]["type"], "text")
        self.assertEqual(blocks[1]["type"], "image")
        self.assertEqual(blocks[1]["source"]["type"], "base64")
        self.assertEqual(blocks[1]["source"]["media_type"], "image/png")

    def test_response_translation_matrix(self):
        # 1. Gemini Response -> OpenAI & Anthropic
        gemini_resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Refactoring completed successfully."},
                            {
                                "functionCall": {
                                    "name": "run_tests",
                                    "args": {"filter": "test_auth"},
                                }
                            },
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 20},
        }

        # Gemini -> OpenAI
        openai_resp = translate_response(
            gemini_resp, from_format="gemini", to_format="openai", model="gpt-4o"
        )
        self.assertEqual(
            openai_resp["choices"][0]["message"]["content"], "Refactoring completed successfully."
        )
        tc = openai_resp["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(tc), 1)
        self.assertEqual(tc[0]["function"]["name"], "run_tests")
        self.assertIn('"filter": "test_auth"', tc[0]["function"]["arguments"])
        self.assertEqual(openai_resp["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(openai_resp["usage"]["prompt_tokens"], 50)

        # Gemini -> Anthropic
        anthropic_resp = translate_response(
            gemini_resp, from_format="gemini", to_format="anthropic", model="claude-3-5-sonnet"
        )
        content_blocks = anthropic_resp["content"]
        self.assertEqual(content_blocks[0]["type"], "text")
        self.assertEqual(content_blocks[0]["text"], "Refactoring completed successfully.")
        self.assertEqual(content_blocks[1]["type"], "tool_use")
        self.assertEqual(content_blocks[1]["name"], "run_tests")
        self.assertEqual(content_blocks[1]["input"], {"filter": "test_auth"})
        self.assertEqual(anthropic_resp["stop_reason"], "tool_use")

        # 2. OpenAI Response -> Anthropic & Gemini
        openai_source = {
            "id": "chatcmpl-test",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Analysis ready",
                        "reasoning_content": "Step 1: Check credentials.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "app.py"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40},
        }

        anth_from_oai = translate_response(
            openai_source, from_format="openai", to_format="anthropic"
        )
        self.assertEqual(anth_from_oai["stop_reason"], "tool_use")
        types = [b["type"] for b in anth_from_oai["content"]]
        self.assertIn("thinking", types)
        self.assertIn("text", types)
        self.assertIn("tool_use", types)

        gem_from_oai = translate_response(openai_source, from_format="openai", to_format="gemini")
        gem_parts = gem_from_oai["candidates"][0]["content"]["parts"]
        self.assertTrue(any(p.get("thought") for p in gem_parts))
        self.assertTrue(any("functionCall" in p for p in gem_parts))

    def test_streaming_transformer_gemini_to_openai(self):
        transformer = create_streaming_transformer(
            from_format="gemini",
            to_format="openai",
            model="gpt-4o",
            stream_id="chatcmpl-stream-1",
        )

        # Chunk 1: Text delta
        chunk1 = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello world"}],
                        "role": "model",
                    }
                }
            ]
        }
        out1 = transformer.transform_chunk(chunk1)
        self.assertEqual(len(out1), 1)
        self.assertEqual(out1[0]["choices"][0]["delta"]["content"], "Hello world")

        # Chunk 2: Tool call chunk
        chunk2 = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "bash", "args": {"command": "pwd"}}}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 15},
        }
        out2 = transformer.transform_chunk(chunk2)
        self.assertEqual(len(out2), 1)
        tc_delta = out2[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc_delta["function"]["name"], "bash")
        self.assertEqual(out2[0]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(out2[0]["usage"]["total_tokens"], 25)

    def test_streaming_transformer_gemini_to_anthropic(self):
        transformer = create_streaming_transformer(
            from_format="gemini",
            to_format="anthropic",
            model="claude-3-5-sonnet",
            stream_id="msg-stream-1",
        )

        # Chunk 1: Thinking and text
        chunk1 = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "Thinking process..."},
                            {"text": "Here is the plan."},
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        events1 = transformer.transform_chunk(chunk1)
        types1 = [e["type"] for e in events1]
        self.assertIn("message_start", types1)
        self.assertIn("content_block_start", types1)
        self.assertIn("content_block_delta", types1)

        # Chunk 2: Finish
        chunk2 = {
            "candidates": [
                {
                    "content": {"parts": [], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 30},
        }
        events2 = transformer.transform_chunk(chunk2)
        types2 = [e["type"] for e in events2]
        self.assertIn("message_delta", types2)
        self.assertIn("message_stop", types2)

    def test_custom_protocol_registry(self):
        class DummyProtocolAdapter(BaseProtocolAdapter):
            protocol_name = "dummy"

            def parse_request(self, payload):
                return CanonicalRequest(model="dummy-m")

            def render_request(self, req):
                return {"dummy_model": req.model}

            def parse_response(self, payload, model=None):
                return CanonicalResponse(content="dummy", model="dummy-m")

            def render_response(self, resp):
                return {"dummy_out": resp.content}

            def parse_chunk(self, chunk, state):
                return CanonicalChunk(text_delta="dummy")

            def render_chunk(self, chunk, state):
                return [{"dummy_chunk": chunk.text_delta}]

        ProtocolRegistry.register("dummy", DummyProtocolAdapter())
        self.assertIn("dummy", ProtocolRegistry.list_protocols())

        out = translate_request({"model": "test"}, from_format="openai", to_format="dummy")
        self.assertEqual(out["dummy_model"], "test")

    def test_provider_alias_resolutions(self):
        # Claude -> Copilot / Microsoft
        anthropic_payload = {
            "model": "claude-3-7-sonnet",
            "messages": [{"role": "user", "content": "Write quicksort"}],
        }
        copilot_req = translate_request(
            anthropic_payload, from_format="claude", to_format="copilot"
        )
        self.assertEqual(copilot_req["model"], "claude-3-7-sonnet")
        self.assertEqual(copilot_req["messages"][0]["content"], "Write quicksort")

        # OpenCode / Cursor -> Antigravity / Google
        cursor_payload = {
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": "Explain AST"}],
        }
        antigravity_req = translate_request(
            cursor_payload, from_format="opencode", to_format="antigravity"
        )
        self.assertIn("contents", antigravity_req)
        self.assertEqual(antigravity_req["contents"][0]["parts"][0]["text"], "Explain AST")

        # Custom alias registration
        ProtocolRegistry.register_alias("my_custom_gateway", "openai")
        self.assertEqual(ProtocolRegistry.resolve_protocol("my_custom_gateway"), "openai")
        custom_req = translate_request(
            anthropic_payload, from_format="claude", to_format="my_custom_gateway"
        )
        self.assertEqual(custom_req["messages"][0]["role"], "user")

        # Helpful actionable error on unknown provider
        with self.assertRaises(ValueError) as ctx:
            translate_request(
                anthropic_payload, from_format="claude", to_format="unknown_service_xyz"
            )
        err_msg = str(ctx.exception)
        self.assertIn("Unknown protocol or provider: 'unknown_service_xyz'", err_msg)
        self.assertIn("ProtocolRegistry.register_alias", err_msg)

        # Reset aliases back to default
        ProtocolRegistry.reset_aliases()

    def test_error_response_translation(self):
        openai_err = {
            "error": {"message": "Rate limit exceeded", "type": "rate_limit_error", "code": 429}
        }
        anth_err = translate_response(openai_err, from_format="openai", to_format="anthropic")
        self.assertEqual(anth_err["type"], "error")
        self.assertEqual(anth_err["error"]["code"], 429)

        gem_err = translate_response(anth_err, from_format="anthropic", to_format="gemini")
        self.assertIn("error", gem_err)
        self.assertEqual(gem_err["error"]["code"], 429)

    def test_transform_sse_line_gemini_to_openai(self):
        transformer = create_streaming_transformer("gemini", "openai", model="gpt-4o")
        sse_input = (
            'data: {"candidates": [{"content": {"parts": [{"text": "Streaming chunk"}]}}]}\n\n'
        )
        out_lines = transformer.transform_sse_line(sse_input)
        self.assertEqual(len(out_lines), 1)
        self.assertTrue(out_lines[0].startswith("data: {"))
        self.assertIn('"content": "Streaming chunk"', out_lines[0])

        # Test keepalive / comment line
        keepalive = ": ping\n\n"
        self.assertEqual(transformer.transform_sse_line(keepalive), [keepalive])

    def test_backwards_compatible_universal_adapter_class_methods(self):
        openai_req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        ir = UniversalProtocolAdapter.request_from_openai(openai_req)
        self.assertEqual(ir.model, "gpt-4o")

        gemini_dict = UniversalProtocolAdapter.request_to_gemini(ir)
        self.assertIn("contents", gemini_dict)

        gemini_resp = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "hello back"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ]
        }
        resp_ir = UniversalProtocolAdapter.response_from_gemini(gemini_resp, model="gemini-2.5-pro")
        self.assertEqual(resp_ir.content, "hello back")

        openai_resp = UniversalProtocolAdapter.response_to_openai(resp_ir)
        self.assertEqual(openai_resp["choices"][0]["message"]["content"], "hello back")


class TestToolSchemaSanitizerAndMCPBridge(unittest.TestCase):
    """Tests for Tool Schema Sanitization (Gemini/OpenAI), MCP JSON-RPC Bridges, and A2A Governance."""

    def test_sanitize_tool_schema_for_gemini(self):
        complex_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "title": "SearchConfig",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "default": "python",
                    "examples": ["machine learning", "ast-grep"],
                },
                "location": {
                    "$ref": "#/$defs/Location",
                },
                "limit": {
                    "anyOf": [
                        {"type": "integer", "description": "Max items to return"},
                        {"type": "null"},
                    ]
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "$ref": "#/$defs/Tag",
                    },
                },
            },
            "required": ["query"],
            "$defs": {
                "Location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "default": "Berlin"},
                        "country": {"type": "string"},
                    },
                    "required": ["city"],
                },
                "Tag": {
                    "type": "string",
                    "examples": ["ai", "rust"],
                },
            },
        }

        sanitized = sanitize_tool_schema_for_gemini(complex_schema)

        # Forbidden keys removed
        self.assertNotIn("$schema", sanitized)
        self.assertNotIn("$defs", sanitized)
        self.assertNotIn("additionalProperties", sanitized)

        # Property checks
        props = sanitized["properties"]
        self.assertNotIn("default", props["query"])
        self.assertNotIn("examples", props["query"])
        self.assertEqual(props["query"]["type"], "string")

        # Ref resolved
        self.assertNotIn("$ref", props["location"])
        self.assertEqual(props["location"]["type"], "object")
        self.assertIn("city", props["location"]["properties"])
        self.assertNotIn("default", props["location"]["properties"]["city"])

        # anyOf flattened to primary branch
        self.assertNotIn("anyOf", props["limit"])
        self.assertEqual(props["limit"]["type"], "integer")
        self.assertEqual(props["limit"]["description"], "Max items to return")

        # Array items ref resolved
        self.assertNotIn("$ref", props["tags"]["items"])
        self.assertEqual(props["tags"]["items"]["type"], "string")
        self.assertNotIn("examples", props["tags"]["items"])

    def test_sanitize_tool_schema_for_openai_strict(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                    },
                },
            },
            "required": ["name"],
        }

        strict_schema = sanitize_tool_schema_for_openai(schema, strict=True)

        self.assertIs(strict_schema["additionalProperties"], False)
        # All properties must be in required in strict mode
        self.assertEqual(sorted(strict_schema["required"]), ["address", "age", "name"])
        # Nested object must also be strict
        nested = strict_schema["properties"]["address"]
        self.assertIs(nested["additionalProperties"], False)
        self.assertEqual(nested["required"], ["street"])

    def test_convert_tool_definitions_mcp_to_gemini_and_anthropic(self):
        mcp_tools = [
            {
                "name": "lookup_symbol",
                "description": "Looks up a code symbol",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "default": "main"},
                    },
                    "required": ["symbol"],
                },
            }
        ]

        # MCP -> Gemini
        gemini_tools = convert_tool_definitions(mcp_tools, from_format="mcp", to_format="gemini")
        self.assertEqual(len(gemini_tools), 1)
        decls = gemini_tools[0]["function_declarations"]
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0]["name"], "lookup_symbol")
        self.assertEqual(decls[0]["description"], "Looks up a code symbol")
        # Ensure Gemini sanitization occurred (no 'default' in parameters)
        self.assertNotIn("default", decls[0]["parameters"]["properties"]["symbol"])

        # MCP -> Anthropic
        anthropic_tools = convert_tool_definitions(
            mcp_tools, from_format="mcp", to_format="anthropic"
        )
        self.assertEqual(len(anthropic_tools), 1)
        self.assertEqual(anthropic_tools[0]["name"], "lookup_symbol")
        self.assertIn("input_schema", anthropic_tools[0])

        # Anthropic -> MCP
        mcp_back = convert_tool_definitions(
            anthropic_tools, from_format="anthropic", to_format="mcp"
        )
        self.assertEqual(len(mcp_back), 1)
        self.assertEqual(mcp_back[0]["name"], "lookup_symbol")
        self.assertIn("inputSchema", mcp_back[0])

    def test_convert_tool_calls_from_raw_model_output(self):
        raw_output = (
            "Thinking: I should check current directory files.\n"
            "<tool_call>\n"
            '{"name": "run_shell", "arguments": {"command": "ls -la /tmp"}}\n'
            "</tool_call>"
        )

        # Raw -> MCP JSON-RPC call
        mcp_calls = convert_tool_calls(raw_output, from_format="raw", to_format="mcp")
        self.assertEqual(len(mcp_calls), 1)
        self.assertEqual(mcp_calls[0]["method"], "tools/call")
        self.assertEqual(mcp_calls[0]["params"]["name"], "run_shell")
        self.assertEqual(mcp_calls[0]["params"]["arguments"], {"command": "ls -la /tmp"})

        # Raw -> OpenAI tool_calls
        openai_calls = convert_tool_calls(raw_output, from_format="raw", to_format="openai")
        self.assertEqual(len(openai_calls), 1)
        self.assertEqual(openai_calls[0]["type"], "function")
        self.assertEqual(openai_calls[0]["function"]["name"], "run_shell")
        import json

        self.assertEqual(
            json.loads(openai_calls[0]["function"]["arguments"]), {"command": "ls -la /tmp"}
        )

        # Raw -> Anthropic tool_use
        anthropic_calls = convert_tool_calls(raw_output, from_format="raw", to_format="anthropic")
        self.assertEqual(len(anthropic_calls), 1)
        self.assertEqual(anthropic_calls[0]["type"], "tool_use")
        self.assertEqual(anthropic_calls[0]["name"], "run_shell")
        self.assertEqual(anthropic_calls[0]["input"], {"command": "ls -la /tmp"})

    def test_convert_tool_results_across_ecosystems(self):
        mcp_result = {
            "content": [{"type": "text", "text": "Files listed successfully."}],
            "isError": False,
        }

        # MCP -> Anthropic
        anth_results = convert_tool_results(
            mcp_result, from_format="mcp", to_format="anthropic", default_tool_call_id="call_abc"
        )
        self.assertEqual(len(anth_results), 1)
        self.assertEqual(anth_results[0]["type"], "tool_result")
        self.assertEqual(anth_results[0]["tool_use_id"], "call_abc")
        self.assertFalse(anth_results[0]["is_error"])

        # Anthropic -> Gemini
        gem_parts = convert_tool_results(
            anth_results,
            from_format="anthropic",
            to_format="gemini",
            name_map={"call_abc": "run_shell"},
        )
        self.assertEqual(len(gem_parts), 1)
        self.assertIn("functionResponse", gem_parts[0])
        self.assertEqual(gem_parts[0]["functionResponse"]["name"], "run_shell")

        # Gemini -> OpenAI
        openai_tool_msgs = convert_tool_results(gem_parts, from_format="gemini", to_format="openai")
        self.assertEqual(len(openai_tool_msgs), 1)
        self.assertEqual(openai_tool_msgs[0]["role"], "tool")
        self.assertEqual(openai_tool_msgs[0]["name"], "run_shell")

    def test_to_mcp_tool_call_and_result_convenience_helpers(self):
        # From raw string
        raw = '<tool_call>{"name": "fetch", "arguments": {"url": "https://api.github.com"}}</tool_call>'
        mcp_call = to_mcp_tool_call(raw)
        self.assertEqual(mcp_call["method"], "tools/call")
        self.assertEqual(mcp_call["params"]["name"], "fetch")
        self.assertEqual(mcp_call["params"]["arguments"]["url"], "https://api.github.com")

        # From OpenAI dictionary
        openai_dict = {
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
        }
        mcp_call2 = to_mcp_tool_call(openai_dict)
        self.assertEqual(mcp_call2["params"]["name"], "read_file")
        self.assertEqual(mcp_call2["params"]["arguments"]["path"], "README.md")

        # Result helpers
        res1 = to_mcp_call_result({"status": "ok"})
        self.assertEqual(res1["content"][0]["type"], "text")
        self.assertIn('"status": "ok"', res1["content"][0]["text"])
        self.assertFalse(res1["isError"])

        res2 = to_mcp_call_result("Operation timed out", is_error=True)
        self.assertEqual(res2["content"][0]["text"], "Operation timed out")
        self.assertTrue(res2["isError"])

    def test_tool_metadata_and_a2a_governance(self):
        meta = ToolMetadata(
            owner_agent="inboost-agent-coordinator",
            security_level="restricted",
            requires_approval=True,
            category="mutation",
            is_mutation=True,
            is_verification=False,
            idempotent=False,
            custom_annotations={"a2a-protocol": "v1.2", "x-inboost-tenant": "acme"},
        )

        d = meta.to_dict()
        self.assertEqual(d["owner_agent"], "inboost-agent-coordinator")
        self.assertEqual(d["security_level"], "restricted")
        self.assertTrue(d["requires_approval"])
        self.assertEqual(d["annotations"]["a2a-protocol"], "v1.2")

        # Roundtrip
        restored = ToolMetadata.from_dict(d)
        self.assertEqual(restored.owner_agent, meta.owner_agent)
        self.assertEqual(restored.security_level, meta.security_level)
        self.assertEqual(restored.requires_approval, meta.requires_approval)
        self.assertEqual(restored.is_mutation, meta.is_mutation)
        self.assertEqual(restored.custom_annotations["x-inboost-tenant"], "acme")

        # ToolDefinition integration
        tool_def = ToolDefinition(
            name="delete_database_cluster",
            description="Deletes a production database cluster",
            parameters={"type": "object", "properties": {"cluster_id": {"type": "string"}}},
            metadata=meta,
        )
        self.assertTrue(tool_def.metadata.requires_approval)
        self.assertEqual(tool_def.metadata.category, "mutation")

        # A2A Headers encoding / decoding
        headers = meta.to_headers(prefix="x-a2a-")
        self.assertEqual(headers["x-a2a-caller-agent"], "inboost-agent-coordinator")
        self.assertEqual(headers["x-a2a-requires-approval"], "true")
        self.assertEqual(headers["x-a2a-is-mutation"], "true")
        self.assertEqual(headers["x-a2a-security-level"], "restricted")
        self.assertEqual(headers["x-a2a-meta-a2a-protocol"], "v1.2")

        header_restored = ToolMetadata.from_headers(headers, prefix="x-a2a-")
        self.assertEqual(header_restored.caller_agent, "inboost-agent-coordinator")
        self.assertTrue(header_restored.requires_approval)
        self.assertTrue(header_restored.is_mutation)
        self.assertEqual(header_restored.security_level, "restricted")
        self.assertEqual(header_restored.custom_annotations["a2a-protocol"], "v1.2")

    def test_normalize_to_ir_and_auto_detection(self):
        from agent_tool_parser import CanonicalToolCall, ToolCall

        # 1. From native ToolCall
        tc = ToolCall(name="get_weather", args={"city": "Tokyo"})
        ir1 = normalize_to_ir(tc)
        self.assertIsInstance(ir1, CanonicalToolCall)
        self.assertEqual(ir1.name, "get_weather")
        self.assertEqual(ir1.arguments, {"city": "Tokyo"})
        self.assertTrue(ir1.id.startswith("call_get_weather"))

        # 2. From CanonicalToolCall (idempotence)
        ir2 = normalize_to_ir(ir1)
        self.assertEqual(ir2, ir1)

        # 3. From OpenAI dict
        openai_dict = {
            "id": "call_oai_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "quantum"}'},
        }
        ir3 = normalize_to_ir(openai_dict)
        self.assertIsInstance(ir3, CanonicalToolCall)
        self.assertEqual(ir3.name, "search")
        self.assertEqual(ir3.arguments, {"q": "quantum"})
        self.assertEqual(ir3.id, "call_oai_1")

        # 4. From Anthropic dict
        anth_dict = {
            "type": "tool_use",
            "id": "toolu_abc_1",
            "name": "read_doc",
            "input": {"id": 42},
        }
        ir4 = normalize_to_ir(anth_dict)
        self.assertIsInstance(ir4, CanonicalToolCall)
        self.assertEqual(ir4.name, "read_doc")
        self.assertEqual(ir4.arguments, {"id": 42})
        self.assertEqual(ir4.id, "toolu_abc_1")

        # 5. From Gemini dict
        gem_dict = {
            "functionCall": {
                "name": "lookup",
                "args": {"key": "api_token"},
            }
        }
        ir5 = normalize_to_ir(gem_dict)
        self.assertIsInstance(ir5, CanonicalToolCall)
        self.assertEqual(ir5.name, "lookup")
        self.assertEqual(ir5.arguments, {"key": "api_token"})

        # 6. From MCP dict
        mcp_dict = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "fetch_url",
                "arguments": {"url": "https://example.com"},
            },
        }
        ir6 = normalize_to_ir(mcp_dict)
        self.assertIsInstance(ir6, CanonicalToolCall)
        self.assertEqual(ir6.name, "fetch_url")
        self.assertEqual(ir6.arguments, {"url": "https://example.com"})

        # 7. From raw LLM text
        raw_text = '<tool_call>{"name": "exec_cmd", "arguments": {"cmd": "ls -la"}}</tool_call>'
        ir7 = normalize_to_ir(raw_text)
        self.assertIsInstance(ir7, list)
        self.assertEqual(len(ir7), 1)
        self.assertEqual(ir7[0].name, "exec_cmd")
        self.assertEqual(ir7[0].arguments, {"command": "ls -la"})

        # 8. List of mixed inputs
        mixed_list = [tc, openai_dict, anth_dict]
        ir_list = normalize_to_ir(mixed_list)
        self.assertIsInstance(ir_list, list)
        self.assertEqual(len(ir_list), 3)
        self.assertEqual([c.name for c in ir_list], ["get_weather", "search", "read_doc"])

    def test_auto_detection_in_converters(self):
        # convert_tool_definitions with auto detection
        defs = [
            {"name": "mcp_t", "description": "MCP", "inputSchema": {"type": "object"}},
            {
                "type": "function",
                "function": {"name": "oai_t", "description": "OAI", "parameters": {}},
            },
            {"name": "anth_t", "description": "Anth", "input_schema": {}},
        ]
        converted_defs = convert_tool_definitions(defs, to_format="anthropic")
        self.assertEqual(len(converted_defs), 3)
        self.assertTrue(all("input_schema" in d for d in converted_defs))

        # convert_tool_calls with auto detection
        calls = [
            {"functionCall": {"name": "gemini_call", "args": {"x": 10}}},
            {"type": "tool_use", "id": "t1", "name": "anth_call", "input": {"y": 20}},
        ]
        converted_calls = convert_tool_calls(calls, to_format="mcp")
        self.assertEqual(len(converted_calls), 2)
        self.assertEqual(converted_calls[0]["method"], "tools/call")
        self.assertEqual(converted_calls[0]["params"]["name"], "gemini_call")
        self.assertEqual(converted_calls[1]["params"]["name"], "anth_call")

        # convert_tool_results with auto detection
        results = [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "Done"},
            {"functionResponse": {"name": "call_2", "response": {"result": "ok"}}},
        ]
        converted_res = convert_tool_results(results, to_format="openai")
        self.assertEqual(len(converted_res), 2)
        self.assertTrue(all(r["role"] == "tool" for r in converted_res))


if __name__ == "__main__":
    unittest.main()
