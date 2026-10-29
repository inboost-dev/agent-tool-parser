# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

import unittest

from agent_tool_parser import (
    ToolCall,
    ToolCategory,
    categorize_tool,
    is_execution_tool,
    is_mutation_tool,
    is_read_tool,
    is_search_tool,
    is_submit_tool,
    is_verification_tool,
    normalize_tool_args,
    normalize_tool_name,
)


class TestSemantics(unittest.TestCase):
    def test_categorize_tool_mutation(self):
        for name in (
            "Edit",
            "edit",
            "write",
            "str_replace",
            "write_file",
            "edit_file",
            "notebookedit",
            "apply_diff",
        ):
            self.assertEqual(categorize_tool(name), ToolCategory.EDIT)
            self.assertTrue(is_mutation_tool(name))
            self.assertFalse(is_execution_tool(name))

    def test_categorize_tool_execution(self):
        self.assertEqual(categorize_tool("bash"), ToolCategory.EXEC)
        self.assertEqual(categorize_tool("execute_bash"), ToolCategory.EXEC)
        self.assertTrue(is_execution_tool("bash"))
        self.assertFalse(is_mutation_tool("bash"))

    def test_categorize_bash_with_subcommands(self):
        # Test command in bash
        pytest_args = {"command": "pytest tests/test_core.py"}
        self.assertEqual(categorize_tool("bash", pytest_args), ToolCategory.TEST)
        self.assertTrue(is_verification_tool("bash", pytest_args))
        self.assertFalse(is_mutation_tool("bash", pytest_args))

        # Mutation command in bash
        patch_args = {"command": "git apply patch.diff"}
        self.assertEqual(categorize_tool("bash", patch_args), ToolCategory.EDIT)
        self.assertTrue(is_mutation_tool("bash", patch_args))

        redirect_args = {"command": "echo 'fix' > file.py"}
        self.assertEqual(categorize_tool("bash", redirect_args), ToolCategory.EDIT)
        self.assertTrue(is_mutation_tool("bash", redirect_args))

    def test_categorize_search_and_read(self):
        for name in ("grep", "find", "file_search", "list_dir"):
            self.assertEqual(categorize_tool(name), ToolCategory.SEARCH)
            self.assertTrue(is_search_tool(name))

        for name in ("read", "view_file", "cat", "head"):
            self.assertEqual(categorize_tool(name), ToolCategory.READ)
            self.assertTrue(is_read_tool(name))

    def test_categorize_submit(self):
        for name in ("submit", "finish", "complete"):
            self.assertEqual(categorize_tool(name), ToolCategory.SUBMIT)
            self.assertTrue(is_submit_tool(name))

    def test_normalize_tool_name(self):
        self.assertEqual(normalize_tool_name("Edit"), "edit")
        self.assertEqual(normalize_tool_name("  EXECUTE_BASH  "), "bash")
        self.assertEqual(normalize_tool_name("write_file"), "write_file")

    def test_normalize_tool_args(self):
        raw_args = {
            "file_path": "src/main.py",
            "old_string": "def old():",
            "new_string": "def new():",
            "cmd": "ls -la",
        }
        normalized = normalize_tool_args(raw_args)
        self.assertEqual(
            normalized,
            {
                "path": "src/main.py",
                "old_str": "def old():",
                "new_str": "def new():",
                "command": "ls -la",
            },
        )

    def test_tool_call_semantic_properties(self):
        tc = ToolCall(
            name="Edit", args={"file_path": "src/app.py", "old_string": "a", "new_string": "b"}
        )
        self.assertEqual(tc.category, ToolCategory.EDIT)
        self.assertTrue(tc.is_mutation)
        self.assertFalse(tc.is_verification)
        self.assertFalse(tc.is_execution)
        self.assertEqual(tc.normalized_name, "edit")
        self.assertEqual(tc.normalized_args["path"], "src/app.py")
        self.assertEqual(tc.normalized_args["old_str"], "a")

        tc_test = ToolCall(name="bash", args={"command": "pytest -v"})
        self.assertEqual(tc_test.category, ToolCategory.TEST)
        self.assertTrue(tc_test.is_verification)
        self.assertFalse(tc_test.is_mutation)
