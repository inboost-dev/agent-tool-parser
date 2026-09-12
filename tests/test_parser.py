# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

import time
import unittest

from agent_tool_parser import (
    ToolError,
    ToolParser,
    parse_tool_call,
    parse_tool_calls,
    try_parse_tool_calls,
)


class TestAgentToolParser(unittest.TestCase):
    def test_dsml_unquoted_and_truncated(self):
        text = """<｜DSML｜tool_calls>
<｜DSML｜tool name=search>
<｜DSML｜parameter name=pattern string=true>ignore-paths</｜DSML｜parameter>
<｜DSML｜parameter name=path string=true>.</｜DSML｜parameter>
</｜DSML｜invoca"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "search")
        self.assertEqual(call.args, {"pattern": "ignore-paths", "path": "."})

    def test_dsml_quoted_standard(self):
        text = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="read_file">
<｜DSML｜parameter name="path">src/main.py</｜DSML｜parameter>
<｜DSML｜parameter name="offset">10</｜DSML｜parameter>
<｜DSML｜parameter name="limit">50</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "src/main.py", "offset": 10, "limit": 50})

    def test_dsml_colon_syntax(self):
        text = """<｜DSML｜call:bash>
<｜DSML｜parameter name="cmd">git status</｜DSML｜parameter>
</｜DSML｜call:bash>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "git status"})

    def test_dsml_parameter_aliases(self):
        text = """<｜DSML｜tool name=str_replace>
<｜DSML｜parameter name=file>app.py</｜DSML｜parameter>
<｜DSML｜parameter name=old>foo</｜DSML｜parameter>
<｜DSML｜parameter name=new>bar</｜DSML｜parameter>
</｜DSML｜tool>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "str_replace")
        self.assertEqual(call.args, {"path": "app.py", "old_str": "foo", "new_str": "bar"})

    def test_dsml_unclosed_parameters(self):
        text = """<|DSML|tool_calls>
<|DSML|tool name="search">
<|DSML|parameter name=pattern>hello world
<|DSML|parameter name=path>src"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "search")
        self.assertEqual(call.args, {"pattern": "hello world", "path": "src"})

    def test_deepseek_v3_native_tokens(self):
        text = '<｜tool calls begin｜><｜tool call begin｜>function=search<｜tool sep｜>{"pattern": "ignore-paths", "path": "."}<｜tool call end｜><｜tool calls end｜>'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "search")
        self.assertEqual(call.args, {"pattern": "ignore-paths", "path": "."})

    def test_deepseek_v3_markdown_block(self):
        text = """<｜tool calls begin｜><｜tool call begin｜>function=str_replace<｜tool sep｜>```json
{"path": "foo.py", "old_str": "a", "new_str": "b"}
```<｜tool call end｜>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "str_replace")
        self.assertEqual(call.args, {"path": "foo.py", "old_str": "a", "new_str": "b"})

    def test_deepseek_v3_truncated(self):
        text = '<｜tool calls begin｜><｜tool call begin｜>function=read_file<｜tool sep｜>{"path": "file.txt", "offset": 1, "limit": 100}'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "file.txt", "offset": 1, "limit": 100})

    def test_hermes_xml_tool_call(self):
        text = """<tool_call>
{"name": "search", "arguments": {"query": "parse_tool", "path": "src"}}
</tool_call>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "search")
        self.assertEqual(call.args, {"pattern": "parse_tool", "path": "src"})

    def test_hermes_direct_tag_format(self):
        text = "<tool_name>search</tool_name><pattern>test_foo</pattern><path>tests/</path>"
        call = parse_tool_call(text)
        self.assertEqual(call.name, "search")
        self.assertEqual(call.args, {"pattern": "test_foo", "path": "tests/"})

    def test_cdata_code_preservation(self):
        code = "    def test():\n        x = 1 < 2 && 3 > 0\n        return x"
        text = f"""<invoke name="write_file">
<parameter name="path">test.py</parameter>
<parameter name="content"><![CDATA[
{code}
]]></parameter>
</invoke>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.args["content"], code)

    def test_cdata_unclosed_stream(self):
        code = "def foo():\n    print('bar')"
        text = f"""<invoke name="write_file">
<parameter name="path">test.py</parameter>
<parameter name="content"><![CDATA[
{code}"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.args["content"], code)

    def test_thinking_block_stripping(self):
        text = """<think>
I need to inspect the code before proposing an edit.
First I will search for the definition of parse_args.
</think>
```json
{"tool": "search", "pattern": "def parse_args", "path": "src"}
```"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "search")
        self.assertEqual(call.args, {"pattern": "def parse_args", "path": "src"})

    def test_dirty_json_repair(self):
        text = "{'tool': 'bash', 'command': 'ls -la',}"
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "ls -la"})

    def test_conversational_embedded_json(self):
        text = 'Sure, here is the command to run:\n{"tool": "bash", "command": "pytest -q"}\nLet me know if it fails.'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "pytest -q"})

    def test_tool_aliasing(self):
        text = '{"tool": "view", "file": "README.md"}'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "README.md"})

    def test_custom_tool_and_param_aliases(self):
        parser = ToolParser(
            allowed_tools={"lookup", "run"},
            tool_aliases={"find": "lookup", "exec": "run"},
            param_aliases={"target": "key"},
        )
        call = parser.parse('{"tool": "find", "target": "user_id"}')
        self.assertEqual(call.name, "lookup")
        self.assertEqual(call.args, {"key": "user_id"})

    def test_allowed_tools_enforcement(self):
        parser = ToolParser(allowed_tools={"search", "bash"})
        with self.assertRaises(ToolError):
            parser.parse('{"tool": "write_file", "path": "x.py", "content": ""}')

        self.assertIsNone(parser.try_parse('{"tool": "write_file", "path": "x.py", "content": ""}'))

    def test_shell_fallback(self):
        parser = ToolParser(allow_shell_fallback=True)
        call = parser.parse("$ pytest -v tests/test_core.py")
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "pytest -v tests/test_core.py"})

    def test_anthropic_tool_use_json(self):
        text = """<tool_use>
<name>read_file</name>
<arguments>
{"path": "src/agent_tool_parser/parser.py", "offset": 10}
</arguments>
</tool_use>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "src/agent_tool_parser/parser.py", "offset": 10})

    def test_anthropic_tool_use_xml_params(self):
        text = """<tool_use>
<tool_name>read_file</tool_name>
<parameters>
<path>src/main.py</path>
<offset>25</offset>
</parameters>
</tool_use>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "src/main.py", "offset": 25})

    def test_anthropic_ant_tool_use_and_cdata(self):
        text = """<ant_tool_use>
<name>write_file</name>
<arguments>
<path>config.py</path>
<content><![CDATA[
DEBUG = True
SECRET = "test"
]]></content>
</arguments>
</ant_tool_use>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "write_file")
        self.assertEqual(call.args["path"], "config.py")
        self.assertEqual(call.args["content"], 'DEBUG = True\nSECRET = "test"')

    def test_anthropic_streaming_unclosed(self):
        text = """<tool_use>
<name>search</name>
<arguments>
{"query": "def parse_tool_call"
"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "search")
        self.assertEqual(call.args, {"pattern": "def parse_tool_call"})

    def test_openai_tool_calls_nested(self):
        text = """{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\\"path\\": \\"src/main.py\\", \\"offset\\": 10}"
      }
    }
  ]
}"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "src/main.py", "offset": 10})

    def test_openai_function_object(self):
        text = """{
  "type": "function",
  "function": {
    "name": "bash",
    "arguments": {"cmd": "pytest -v"}
  }
}"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "pytest -v"})

    def test_openai_function_call_wrapper(self):
        text = """{
  "function_call": {
    "name": "search",
    "arguments": "{\\"pattern\\": \\"def test\\"}"
  }
}"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "search")
        self.assertEqual(call.args, {"pattern": "def test"})

    def test_openai_array_format(self):
        text = """[
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "arguments": {"file": "README.md"}
    }
  }
]"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "README.md"})

    def test_openai_unescaped_control_characters(self):
        text = '{"tool": "bash", "command": "echo line 1\necho line 2\t# tab"}'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "echo line 1\necho line 2\t# tab"})

    def test_qwen_agent_tokens(self):
        text = """<|action_start|><|action_name|>read_file<|action_args|>{"path": "src/core.py"}<|action_end|>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "src/core.py"})

    def test_qwen_agent_truncated(self):
        text = """<|action_start|><|action_name|>bash<|action_args|>{"cmd": "git status"}"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "git status"})

    def test_hermes_stringified_arguments(self):
        text = """<tool_call>
{"name": "search", "arguments": "{\\"query\\": \\"find_me\\"}"}
</tool_call>"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "search")
        self.assertEqual(call.args, {"pattern": "find_me"})

    def test_chatglm_function_format(self):
        text = """✿FUNCTION✿: bash
✿ARGS✿: {"cmd": "python setup.py build"}"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "python setup.py build"})

    def test_llama_python_tag_expression(self):
        text = """<|python_tag|>bash.call(command="pytest -q")"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "pytest -q"})

    def test_llama_python_tag_json(self):
        text = """<|python_tag|>{"name": "search", "parameters": {"pattern": "def hello"}}"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "search")
        self.assertEqual(call.args, {"pattern": "def hello"})

    def test_python_expr_ast_parsing(self):
        text = 'read_file(path="src/parser.py", offset=15)'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "src/parser.py", "offset": 15})

    def test_python_expr_positional_arg(self):
        text = 'bash("git diff")'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "git diff"})

    def test_mistral_tool_calls(self):
        text = '[TOOL_CALLS] [{"name": "read_file", "arguments": {"path": "README.md"}}]'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.args, {"path": "README.md"})

    def test_react_format_plain(self):
        text = """Thought: Let's run unit tests.
Action: bash
Action Input: pytest tests/test_parser.py"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "pytest tests/test_parser.py"})

    def test_react_format_json(self):
        text = """Action: search
Action Input: {"pattern": "class ToolParser", "path": "src"}"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "search")
        self.assertEqual(call.args, {"pattern": "class ToolParser", "path": "src"})

    def test_react_format_code_block(self):
        text = """Action: bash
Action Input: ```bash
git log -n 1
```"""
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args, {"command": "git log -n 1"})

    def test_python_literals_in_json(self):
        text = '{"tool": "bash", "command": "pytest", "verbose": True, "limit": None}'
        call = parse_tool_call(text)
        self.assertEqual(call.name, "bash")
        self.assertEqual(call.args["command"], "pytest")
        self.assertEqual(call.args["verbose"], True)
        self.assertIsNone(call.args["limit"])

    def test_multi_anthropic_tool_use(self):
        text = """<tool_use>
<name>read_file</name>
<arguments>{"path": "a.py"}</arguments>
</tool_use>
<tool_use>
<name>read_file</name>
<arguments>{"path": "b.py"}</arguments>
</tool_use>"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].args, {"path": "a.py"})
        self.assertEqual(calls[1].name, "read_file")
        self.assertEqual(calls[1].args, {"path": "b.py"})

        # parse_tool_call returns first call
        first = parse_tool_call(text)
        self.assertEqual(first.name, "read_file")
        self.assertEqual(first.args, {"path": "a.py"})

    def test_multi_openai_tool_calls(self):
        text = """{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {"name": "read_file", "arguments": "{\\"path\\": \\"foo.py\\"}"}
    },
    {
      "id": "call_2",
      "type": "function",
      "function": {"name": "search", "arguments": "{\\"query\\": \\"bar\\"}"}
    }
  ]
}"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].args, {"path": "foo.py"})
        self.assertEqual(calls[1].name, "search")
        self.assertEqual(calls[1].args, {"pattern": "bar"})

    def test_multi_openai_top_level_array(self):
        text = """[
  {"type": "function", "function": {"name": "read_file", "arguments": {"file": "1.py"}}},
  {"type": "function", "function": {"name": "read_file", "arguments": {"file": "2.py"}}}
]"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args, {"path": "1.py"})
        self.assertEqual(calls[1].args, {"path": "2.py"})

    def test_multi_deepseek_native_tokens(self):
        text = '<｜tool calls begin｜><｜tool call begin｜>function=bash<｜tool sep｜>{"cmd": "git status"}<｜tool call end｜><｜tool call begin｜>function=bash<｜tool sep｜>{"cmd": "git diff"}<｜tool call end｜><｜tool calls end｜>'
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "bash")
        self.assertEqual(calls[0].args, {"command": "git status"})
        self.assertEqual(calls[1].name, "bash")
        self.assertEqual(calls[1].args, {"command": "git diff"})

    def test_multi_deepseek_dsml(self):
        text = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="read_file">
<｜DSML｜parameter name="path">x.py</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="write_file">
<｜DSML｜parameter name="path">y.py</｜DSML｜parameter>
<｜DSML｜parameter name="content">hello</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].args, {"path": "x.py"})
        self.assertEqual(calls[1].name, "write_file")
        self.assertEqual(calls[1].args, {"path": "y.py", "content": "hello"})

    def test_multi_qwen_agent_tokens(self):
        text = """<|action_start|><|action_name|>read_file<|action_args|>{"path": "f1.py"}<|action_end|><|action_start|><|action_name|>read_file<|action_args|>{"path": "f2.py"}<|action_end|>"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args, {"path": "f1.py"})
        self.assertEqual(calls[1].args, {"path": "f2.py"})

    def test_multi_hermes_xml(self):
        text = """<tool_call>
{"name": "search", "arguments": {"query": "one"}}
</tool_call>
<tool_call>
{"name": "search", "arguments": {"query": "two"}}
</tool_call>"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args, {"pattern": "one"})
        self.assertEqual(calls[1].args, {"pattern": "two"})

    def test_multi_markdown_json_blocks(self):
        text = """I will run these two commands:
```json
{"tool": "bash", "command": "pytest -k test_1"}
```
And then:
```json
{"tool": "bash", "command": "pytest -k test_2"}
```"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args, {"command": "pytest -k test_1"})
        self.assertEqual(calls[1].args, {"command": "pytest -k test_2"})

    def test_multi_react_actions(self):
        text = """Thought: Check status and tests.
Action: bash
Action Input: git status

Action: bash
Action Input: pytest"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args, {"command": "git status"})
        self.assertEqual(calls[1].args, {"command": "pytest"})

    def test_multi_python_expressions(self):
        text = """
read_file(path="a.py", offset=1)
read_file(path="b.py", offset=2)
"""
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args, {"path": "a.py", "offset": 1})
        self.assertEqual(calls[1].args, {"path": "b.py", "offset": 2})

    def test_try_parse_tool_calls_fallback(self):
        text = "Hello! Just explaining things to you."
        calls = try_parse_tool_calls(text)
        self.assertEqual(calls, [])

        with self.assertRaises(ToolError):
            parse_tool_calls(text)

    def test_performance_sub_millisecond(self):
        text = """<｜DSML｜invoke name="read_file">
<｜DSML｜parameter name="path">src/main.py</｜DSML｜parameter>
<｜DSML｜parameter name="offset">10</｜DSML｜parameter>
<｜DSML｜parameter name="limit">50</｜DSML｜parameter>
</｜DSML｜invoke>"""
        for _ in range(50):
            parse_tool_call(text)

        n = 1000
        start = time.perf_counter()
        for _ in range(n):
            parse_tool_call(text)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / n) * 1000.0
        self.assertLess(avg_ms, 0.5, f"Average parse latency too high: {avg_ms:.4f} ms")

    def test_no_syntax_warnings_on_invalid_escapes(self):
        import warnings

        text = 'pattern = re.compile("\\[a-z\\]")\nother = "\\[math\\]"'
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            calls = try_parse_tool_calls(text)
            syntax_warnings = [w for w in recorded if issubclass(w.category, SyntaxWarning)]
            self.assertEqual(
                len(syntax_warnings),
                0,
                f"Found unexpected SyntaxWarnings: {syntax_warnings}",
            )
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
