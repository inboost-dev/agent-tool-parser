# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

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


def test_categorize_tool_mutation():
    for name in ("Edit", "edit", "write", "str_replace", "write_file", "edit_file", "notebookedit", "apply_diff"):
        assert categorize_tool(name) == ToolCategory.EDIT
        assert is_mutation_tool(name) is True
        assert is_execution_tool(name) is False


def test_categorize_tool_execution():
    assert categorize_tool("bash") == ToolCategory.EXEC
    assert categorize_tool("execute_bash") == ToolCategory.EXEC
    assert is_execution_tool("bash") is True
    assert is_mutation_tool("bash") is False


def test_categorize_bash_with_subcommands():
    # Test command in bash
    pytest_args = {"command": "pytest tests/test_core.py"}
    assert categorize_tool("bash", pytest_args) == ToolCategory.TEST
    assert is_verification_tool("bash", pytest_args) is True
    assert is_mutation_tool("bash", pytest_args) is False

    # Mutation command in bash
    patch_args = {"command": "git apply patch.diff"}
    assert categorize_tool("bash", patch_args) == ToolCategory.EDIT
    assert is_mutation_tool("bash", patch_args) is True

    redirect_args = {"command": "echo 'fix' > file.py"}
    assert categorize_tool("bash", redirect_args) == ToolCategory.EDIT
    assert is_mutation_tool("bash", redirect_args) is True


def test_categorize_search_and_read():
    for name in ("grep", "find", "file_search", "list_dir"):
        assert categorize_tool(name) == ToolCategory.SEARCH
        assert is_search_tool(name) is True

    for name in ("read", "view_file", "cat", "head"):
        assert categorize_tool(name) == ToolCategory.READ
        assert is_read_tool(name) is True


def test_categorize_submit():
    for name in ("submit", "finish", "complete"):
        assert categorize_tool(name) == ToolCategory.SUBMIT
        assert is_submit_tool(name) is True


def test_normalize_tool_name():
    assert normalize_tool_name("Edit") == "edit"
    assert normalize_tool_name("  EXECUTE_BASH  ") == "bash"
    assert normalize_tool_name("write_file") == "write_file"


def test_normalize_tool_args():
    raw_args = {
        "file_path": "src/main.py",
        "old_string": "def old():",
        "new_string": "def new():",
        "cmd": "ls -la",
    }
    normalized = normalize_tool_args(raw_args)
    assert normalized == {
        "path": "src/main.py",
        "old_str": "def old():",
        "new_str": "def new():",
        "command": "ls -la",
    }


def test_tool_call_semantic_properties():
    tc = ToolCall(name="Edit", args={"file_path": "src/app.py", "old_string": "a", "new_string": "b"})
    assert tc.category == ToolCategory.EDIT
    assert tc.is_mutation is True
    assert tc.is_verification is False
    assert tc.is_execution is False
    assert tc.normalized_name == "edit"
    assert tc.normalized_args["path"] == "src/app.py"
    assert tc.normalized_args["old_str"] == "a"

    tc_test = ToolCall(name="bash", args={"command": "pytest -v"})
    assert tc_test.category == ToolCategory.TEST
    assert tc_test.is_verification is True
    assert tc_test.is_mutation is False
