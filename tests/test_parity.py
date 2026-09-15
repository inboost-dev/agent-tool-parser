# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT
#
# Dual-Engine Parity Verification Suite
# Directly compares pure-Python reference implementation against compiled PyO3 native bindings.

from __future__ import annotations

import json
import unittest

import agent_tool_parser.cleaners as py_cleaners
import agent_tool_parser.parser as py_parser
from agent_tool_parser.models import ToolError as PyToolError

try:
    import agent_tool_parser._accelerated as acc

    HAS_ACCELERATED = True
except ImportError:
    HAS_ACCELERATED = False


@unittest.skipUnless(HAS_ACCELERATED, "Compiled native PyO3 bindings (_accelerated) not built")
class TestDualEngineParity(unittest.TestCase):
    """Systematic differential testing ensuring identical behavior between Python and PyO3."""

    def _assert_equivalent(
        self,
        inp: str,
        acc_result: any,
        py_result: any,
        test_context: str,
        acc_err: Exception | None = None,
        py_err: Exception | None = None,
    ) -> None:
        if acc_err is not None or py_err is not None:
            self.assertIsNotNone(
                acc_err,
                f"[{test_context}] Pure-Python raised error {py_err!r} but PyO3 succeeded with {acc_result!r}",
            )
            self.assertIsNotNone(
                py_err,
                f"[{test_context}] PyO3 raised error {acc_err!r} but Pure-Python succeeded with {py_result!r}",
            )
            self.assertIsInstance(
                acc_err,
                (acc.ToolError, PyToolError),
                f"[{test_context}] PyO3 error is not a ToolError: {acc_err!r}",
            )
            self.assertIsInstance(
                py_err,
                (acc.ToolError, PyToolError),
                f"[{test_context}] Pure-Python error is not a ToolError: {py_err!r}",
            )
            return

        if acc_result is None or py_result is None:
            self.assertIsNone(
                acc_result,
                f"[{test_context}] Pure-Python returned None but PyO3 returned {acc_result!r}",
            )
            self.assertIsNone(
                py_result,
                f"[{test_context}] PyO3 returned None but Pure-Python returned {py_result!r}",
            )
            return

        if isinstance(acc_result, list) and isinstance(py_result, list):
            self.assertEqual(
                len(acc_result),
                len(py_result),
                f"[{test_context}] Length mismatch: PyO3 has {len(acc_result)}, Python has {len(py_result)}",
            )
            for i, (ac, pc) in enumerate(zip(acc_result, py_result)):
                self.assertEqual(
                    ac.name,
                    pc.name,
                    f"[{test_context}] Call[{i}] tool name mismatch: PyO3={ac.name!r}, Python={pc.name!r}",
                )
                self.assertEqual(
                    ac.args,
                    pc.args,
                    f"[{test_context}] Call[{i}] tool args mismatch: PyO3={ac.args!r}, Python={pc.args!r}",
                )
            return

        if hasattr(acc_result, "name") and hasattr(py_result, "name"):
            self.assertEqual(
                acc_result.name,
                py_result.name,
                f"[{test_context}] Tool name mismatch: PyO3={acc_result.name!r}, Python={py_result.name!r}",
            )
            self.assertEqual(
                acc_result.args,
                py_result.args,
                f"[{test_context}] Tool args mismatch: PyO3={acc_result.args!r}, Python={py_result.args!r}",
            )
            return

        self.assertEqual(
            acc_result,
            py_result,
            f"[{test_context}] Primitive result mismatch: PyO3={acc_result!r}, Python={py_result!r}",
        )

    def _check_differential(
        self,
        inp: str,
        name_prefix: str,
        allowed_tools: set[str] | None = None,
        param_aliases: dict[str, str] | None = None,
        tool_aliases: dict[str, str] | None = None,
    ) -> None:
        # 1. parse_tool_call / parse
        acc_res, acc_err = None, None
        try:
            if allowed_tools or param_aliases or tool_aliases:
                parser = acc.ToolParser(
                    allowed_tools=allowed_tools,
                    param_aliases=param_aliases,
                    tool_aliases=tool_aliases,
                )
                acc_res = parser.parse(inp)
            else:
                acc_res = acc.parse_tool_call(inp)
        except Exception as e:
            acc_err = e

        py_res, py_err = None, None
        try:
            if allowed_tools or param_aliases or tool_aliases:
                parser = py_parser.ToolParser(
                    allowed_tools=allowed_tools,
                    param_aliases=param_aliases,
                    tool_aliases=tool_aliases,
                )
                py_res = parser.parse(inp)
            else:
                py_res = py_parser.parse_tool_call(inp)
        except Exception as e:
            py_err = e

        self._assert_equivalent(
            inp, acc_res, py_res, f"{name_prefix}::parse_tool_call", acc_err, py_err
        )

        # 2. parse_tool_calls / parse_all
        acc_res, acc_err = None, None
        try:
            if allowed_tools or param_aliases or tool_aliases:
                parser = acc.ToolParser(
                    allowed_tools=allowed_tools,
                    param_aliases=param_aliases,
                    tool_aliases=tool_aliases,
                )
                acc_res = parser.parse_all(inp)
            else:
                acc_res = acc.parse_tool_calls(inp)
        except Exception as e:
            acc_err = e

        py_res, py_err = None, None
        try:
            if allowed_tools or param_aliases or tool_aliases:
                parser = py_parser.ToolParser(
                    allowed_tools=allowed_tools,
                    param_aliases=param_aliases,
                    tool_aliases=tool_aliases,
                )
                py_res = parser.parse_all(inp)
            else:
                py_res = py_parser.parse_tool_calls(inp)
        except Exception as e:
            py_err = e

        self._assert_equivalent(
            inp, acc_res, py_res, f"{name_prefix}::parse_tool_calls", acc_err, py_err
        )

        # 3. try_parse_tool_call
        acc_res = acc.try_parse_tool_call(inp)
        py_res = py_parser.try_parse_tool_call(inp)
        self._assert_equivalent(inp, acc_res, py_res, f"{name_prefix}::try_parse_tool_call")

        # 4. try_parse_tool_calls
        acc_res = acc.try_parse_tool_calls(inp)
        py_res = py_parser.try_parse_tool_calls(inp)
        self._assert_equivalent(inp, acc_res, py_res, f"{name_prefix}::try_parse_tool_calls")

        # 5. strip_thinking
        acc_th = acc.strip_thinking(inp)
        py_th = py_cleaners.strip_thinking(inp)
        self._assert_equivalent(inp, acc_th, py_th, f"{name_prefix}::strip_thinking")

    def test_parity_heterogeneous_formats(self):
        """Parity across OpenAI, Claude, DeepSeek DSML, Qwen, ChatGLM, Llama AST, and ReAct."""
        samples = [
            '{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}',
            '{"tool": "bash", "command": "ls -la /tmp"}',
            '{"tool_calls": [{"type": "function", "function": {"name": "search", "arguments": "{\\"q\\": \\"rust\\"}"}}]}',
            '{"function_call": {"name": "calc", "arguments": "{\\"expr\\": \\"2+2\\"}"}}',
            '[{"name": "fetch", "arguments": {"url": "https://example.com"}}]',
            '<tool_use>\n<name>read_file</name>\n<arguments>\n{"path": "src/main.rs"}\n</arguments>\n</tool_use>',
            '<invoke name="write_file">\n<parameter name="path">test.py</parameter>\n<parameter name="content"><![CDATA[print("hello")]]></parameter>\n</invoke>',
            '<ant_tool_use>\n<name>grep</name>\n<arguments>\n{"pattern": "TODO"}\n</arguments>\n</ant_tool_use>',
            '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="deploy">\n<｜DSML｜parameter name="env">prod</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
            '<｜DSML｜invoke name="query"><｜DSML｜parameter name="sql">SELECT 1;</｜DSML｜parameter></｜DSML｜invoke>',
            "<｜DSML｜invoke:search><｜DSML｜parameter:pattern>regex</｜DSML｜parameter:pattern></｜DSML｜invoke:search>",
            "<｜DSML｜tool name=search><｜DSML｜parameter name=query>vector</｜DSML｜parameter>",
            '<｜tool calls begin｜><｜tool call begin｜>function=bash<｜tool sep｜>{"command": "pytest"}<｜tool call end｜><｜tool calls end｜>',
            '<｜tool call begin｜>function=calc<｜tool sep｜>{"a": 1, "b": 2}<｜tool call end｜>',
            '<|action_start|><|action_name|>execute<|action_args|>{"cmd": "git status"}<|action_end|>',
            '✿FUNCTION✿: execute_code\n✿ARGS✿: {"lang": "python", "code": "2 + 2"}',
            'call:get_weather(city="Paris", temp=22)',
            'call:bash("echo hello")',
            'Thought: I need to check files.\nAction: bash\nAction Input: {"command": "ls"}',
            "Action: search\nAction Input: python docs",
        ]
        for i, sample in enumerate(samples):
            self._check_differential(sample, f"FormatSample[{i}]")

    def test_parity_truncated_and_dirty_inputs(self):
        """Parity on truncated streams, bracket tracking, and heuristic repairs."""
        samples = [
            '{"tool": "bash", "command": "echo hi',
            '{"tool": "bash", "command":',
            '{"tool": "process", "items": ["a", "b", ',
            '<invoke name="save"><parameter name="content"><![CDATA[def func():\n    return True',
            "{'tool': 'bash', 'command': 'echo \"Hello\"'}",
            '{"tool": "test", "active": True, "empty": None, "debug": False,}',
            '<think>contemplating...</think>{"tool": "bash", "command": "pwd"}',
            '<think>maybe call <tool_call>{"name": "rm"}</tool_call> wait no</think> I will not do that.',
            '<thought>reasoning</thought><invoke name="calc"><parameter name="x">10</parameter></invoke>',
        ]
        for i, sample in enumerate(samples):
            self._check_differential(sample, f"TruncatedSample[{i}]")

    def test_parity_degenerate_and_adversarial(self):
        """Parity on pathological inputs, control characters, and collisions."""
        samples = [
            "",
            "   ",
            "\n\n\n",
            '{"tool": null, "command": "ls"}',
            '{"tool": 12345, "command": "ls"}',
            '{"tool": true, "command": "ls"}',
            '{"tool": ["bash"], "command": "ls"}',
            "{}",
            "[]",
            "x < 10 and y > 20",
            "std::vector<std::pair<int, int>> table;",
            "Action: just chatting without Action Input",
            '<invoke name="bash"><parameter name="cmd">\x1b[31mhello\x1b[0m</parameter></invoke>',
            '{"tool": "bash", "command": "echo hello\x00world"}',
            '\ufeff\u200b{"tool": "bash", "command": "ls"}\u200d',
            # MiniMax truncated JSON prefixes
            'tool": "bash", "command": "git diff"}',
            'name": "bash", "input": {"command": "ls"}}',
            'action": "read_file", "path": "main.py"}',
            '"tool": "read_file", "path": "src/app.py", "offset": 1, "limit": 100}',
            '"name": "bash", "command": "pytest"}',
            '": "read_file", "path": "sklearn/impute/_iterative.py", "offset": 1, "limit": 100}',
            '": "search", "pattern": "initial_strategy", "path": "."}',
            '{": "read_file", "path": "sklearn/impute/_iterative.py", "offset": 10, "limit": 50}',
            '{ ": "search", "pattern": "def _discover_files", "path": "pylint"}',
            'Now I will read the target file to inspect the function:\n": "read_file", "path": "a.py"}',
            'Let me check git status:\ntool": "bash", "command": "git status"}',
            '```json\ntool": "bash", "command": "git status"}\n```',
        ]
        for i, sample in enumerate(samples):
            self._check_differential(sample, f"DegenerateSample[{i}]")

    def test_parity_configured_parsers(self):
        """Parity for ToolParser instances with allowed_tools, tool_aliases, and param_aliases."""
        self._check_differential(
            '{"tool": "run", "cmd": "pytest"}',
            "AliasesTest",
            param_aliases={"cmd": "command"},
            tool_aliases={"run": "bash"},
        )
        self._check_differential(
            '{"tool": "disallowed", "cmd": "ls"}',
            "AllowedToolsRejectTest",
            allowed_tools={"bash", "python"},
        )
        self._check_differential(
            '{"tool": "bash", "cmd": "ls"}',
            "AllowedToolsAcceptTest",
            allowed_tools={"bash", "python"},
        )

    def test_parity_parser_method_aliases(self):
        """Verify all method aliases match on ToolParser in both engines."""
        py_p = py_parser.ToolParser()
        acc_p = acc.ToolParser()
        inp = '{"tool": "bash", "command": "pwd"}'

        # parse vs parse
        self.assertEqual(acc_p.parse(inp).name, py_p.parse(inp).name)
        self.assertEqual(acc_p.parse(inp).args, py_p.parse(inp).args)

        # parse_all vs parse_all
        acc_calls = acc_p.parse_all(inp)
        py_calls = py_p.parse_all(inp)
        self.assertEqual(len(acc_calls), len(py_calls))
        self.assertEqual(acc_calls[0].name, py_calls[0].name)

        # parse_calls vs parse_calls
        self.assertEqual(len(acc_p.parse_calls(inp)), len(py_p.parse_calls(inp)))

        # parse_many vs parse_many
        self.assertEqual(len(acc_p.parse_many(inp)), len(py_p.parse_many(inp)))

        # try_parse vs try_parse
        self.assertEqual(acc_p.try_parse(inp).name, py_p.try_parse(inp).name)

        # try_parse_all vs try_parse_all
        self.assertEqual(len(acc_p.try_parse_all(inp)), len(py_p.try_parse_all(inp)))

        # try_parse_calls vs try_parse_calls
        self.assertEqual(len(acc_p.try_parse_calls(inp)), len(py_p.try_parse_calls(inp)))

        # try_parse_many vs try_parse_many
        self.assertEqual(len(acc_p.try_parse_many(inp)), len(py_p.try_parse_many(inp)))

    def test_parity_clean_json_str(self):
        """Parity for clean_json_str heuristic repairs."""
        json_samples = [
            "{'a': 1, 'b': 'hello'}",
            "{'a': 'hello \"world\"'}",
            '{"a": 1, "b": [1, 2, ], }',
            '{"a": True, "b": False, "c": None}',
            '{"valid": 123}',
            "not a json",
            'tool": "bash", "command": "git diff"}',
            '"name": "bash", "command": "pytest"}',
            '": "read_file", "path": "sklearn/impute/_iterative.py", "offset": 1, "limit": 100}',
            '{": "read_file", "path": "sklearn/impute/_iterative.py", "offset": 10, "limit": 50}',
            '{ ": "search", "pattern": "def _discover_files", "path": "pylint"}',
        ]
        for inp in json_samples:
            acc_cj = acc.clean_json_str(inp)
            py_cj = py_cleaners.clean_json_str(inp)
            try:
                self.assertEqual(json.loads(acc_cj), json.loads(py_cj))
            except Exception:
                self.assertEqual(acc_cj, py_cj)

    def test_parity_normalize_truncated_json_prefix(self):
        """Parity for normalize_truncated_json_prefix."""
        samples = [
            'tool": "bash", "command": "git diff"}',
            'name": "bash", "input": {"command": "ls"}}',
            'action": "read_file", "path": "main.py"}',
            '"tool": "read_file", "path": "src/app.py", "offset": 1, "limit": 100}',
            '"name": "bash", "command": "pytest"}',
            '": "read_file", "path": "sklearn/impute/_iterative.py", "offset": 1, "limit": 100}',
            '": "search", "pattern": "initial_strategy", "path": "."}',
            '{": "read_file", "path": "sklearn/impute/_iterative.py", "offset": 10, "limit": 50}',
            '{ ": "search", "pattern": "def _discover_files", "path": "pylint"}',
            ': "read_file", "path": "a.py"}',
            "normal string",
            '{"tool": "bash"}',
        ]
        for inp in samples:
            acc_norm = acc.normalize_truncated_json_prefix(inp)
            py_norm = py_cleaners.normalize_truncated_json_prefix(inp)
            self.assertEqual(acc_norm, py_norm)


if __name__ == "__main__":
    unittest.main()
