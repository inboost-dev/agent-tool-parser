# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

from __future__ import annotations

from enum import Enum
from typing import Any


class ToolCategory(str, Enum):
    """Semantic categorization of agent tool operations across LLM frameworks.

    Provides a unified categorization model across OpenHands, Claude Code,
    Cursor, Aider, and custom InBoost runtime agents.
    """

    EDIT = "edit"  # Modifies code or files on disk
    EXEC = "exec"  # Executes generic shell or bash commands
    READ = "read"  # Inspects file content or directory listings
    SEARCH = "search"  # Searches files, patterns, or AST
    TEST = "test"  # Executes verification test suites or assertion checks
    SUBMIT = "submit"  # Concludes task or submits resolution patch
    L0_ACCEL = "l0_accel"  # Deterministic local accelerators (AST, syntax, log cleaning)
    UNKNOWN = "unknown"  # Unclassified or custom domain tool


CANONICAL_TOOL_MAP: dict[str, tuple[str, ToolCategory]] = {
    # 1. File Modification / Mutation Tools (Claude Code, OpenHands, Aider, Cursor)
    "edit": ("edit", ToolCategory.EDIT),
    "write": ("write", ToolCategory.EDIT),
    "str_replace": ("str_replace", ToolCategory.EDIT),
    "write_file": ("write_file", ToolCategory.EDIT),
    "edit_file": ("edit_file", ToolCategory.EDIT),
    "save_file": ("save_file", ToolCategory.EDIT),
    "replace": ("replace", ToolCategory.EDIT),
    "notebookedit": ("notebookedit", ToolCategory.EDIT),
    "apply_diff": ("apply_diff", ToolCategory.EDIT),
    "create_file": ("write_file", ToolCategory.EDIT),
    "modify_file": ("edit_file", ToolCategory.EDIT),
    "patch_file": ("apply_diff", ToolCategory.EDIT),

    # 2. Command Execution Tools
    "bash": ("bash", ToolCategory.EXEC),
    "execute_bash": ("bash", ToolCategory.EXEC),
    "run_bash": ("bash", ToolCategory.EXEC),
    "terminal": ("bash", ToolCategory.EXEC),
    "cmd": ("bash", ToolCategory.EXEC),
    "execute_command": ("bash", ToolCategory.EXEC),
    "run_shell": ("bash", ToolCategory.EXEC),

    # 3. Read / Inspection Tools
    "read": ("read", ToolCategory.READ),
    "read_file": ("read", ToolCategory.READ),
    "view": ("read", ToolCategory.READ),
    "view_file": ("read", ToolCategory.READ),
    "cat": ("read", ToolCategory.READ),
    "head": ("read", ToolCategory.READ),
    "tail": ("read", ToolCategory.READ),

    # 4. Search Tools
    "grep": ("grep", ToolCategory.SEARCH),
    "grep_search": ("grep", ToolCategory.SEARCH),
    "search": ("search", ToolCategory.SEARCH),
    "file_search": ("file_search", ToolCategory.SEARCH),
    "find": ("find", ToolCategory.SEARCH),
    "glob": ("glob", ToolCategory.SEARCH),
    "list_dir": ("list_dir", ToolCategory.SEARCH),
    "list_directory": ("list_dir", ToolCategory.SEARCH),
    "ls": ("list_dir", ToolCategory.SEARCH),

    # 5. Verification / Testing Tools
    "pytest": ("pytest", ToolCategory.TEST),
    "run_tests": ("run_tests", ToolCategory.TEST),
    "test": ("test", ToolCategory.TEST),
    "l0_run_test": ("l0_run_test", ToolCategory.TEST),

    # 6. Conclude / Submit Tools
    "submit": ("submit", ToolCategory.SUBMIT),
    "finish": ("finish", ToolCategory.SUBMIT),
    "complete": ("complete", ToolCategory.SUBMIT),
    "exit": ("exit", ToolCategory.SUBMIT),

    # 7. Local L0 Acceleration Tools (InBoost LongRun)
    "l0_verify_syntax": ("l0_verify_syntax", ToolCategory.L0_ACCEL),
    "l0_verify_edit_scope": ("l0_verify_edit_scope", ToolCategory.L0_ACCEL),
    "l0_clean_test_log": ("l0_clean_test_log", ToolCategory.L0_ACCEL),
    "l0_view_outline": ("l0_view_outline", ToolCategory.L0_ACCEL),
    "l0_format_diff": ("l0_format_diff", ToolCategory.L0_ACCEL),
}

# Cross-agent argument aliases
ARG_ALIASES: dict[str, str] = {
    "file_path": "path",
    "filepath": "path",
    "filename": "path",
    "target_file": "path",
    "file": "path",
    "old_string": "old_str",
    "old_text": "old_str",
    "old_code": "old_str",
    "new_string": "new_str",
    "new_text": "new_str",
    "new_code": "new_str",
    "cmd": "command",
    "shell_command": "command",
    "query_string": "query",
}

TEST_CMD_PATTERNS = (
    "pytest",
    "python -m unittest",
    "python -m pytest",
    "./tests/runtests.py",
    "python tests/runtests.py",
    "runtests.py",
    "tox",
    "nosetests",
    "ctest",
    "cargo test",
    "go test",
    "npm test",
)

BASH_MUTATION_OPS = (
    ">",
    ">>",
    "tee ",
    "patch",
    "sed -i",
    "git apply",
    "git checkout",
)


def normalize_tool_name(raw_name: str) -> str:
    """Normalize tool name by lowercasing and resolving canonical mappings."""
    if not raw_name:
        return ""
    clean = raw_name.strip().lower()
    if clean in CANONICAL_TOOL_MAP:
        return CANONICAL_TOOL_MAP[clean][0]
    return clean


def normalize_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize argument keys across agent frameworks (e.g. file_path -> path)."""
    if not isinstance(args, dict):
        return args
    normalized: dict[str, Any] = {}
    for k, v in args.items():
        target_k = ARG_ALIASES.get(k, k)
        normalized[target_k] = v
    return normalized


def categorize_tool(
    name: str, args: dict[str, Any] | None = None
) -> ToolCategory:
    """Determine semantic category of a tool invocation.

    For execution tools (bash), inspects command arguments to disambiguate
    test commands (e.g. pytest) and file mutations (e.g. patch, echo >).
    """
    if not name:
        return ToolCategory.UNKNOWN
    clean = name.strip().lower()

    # Direct canonical lookup
    canonical_entry = CANONICAL_TOOL_MAP.get(clean)
    category = canonical_entry[1] if canonical_entry else ToolCategory.UNKNOWN

    # For execution commands, inspect the command string if available
    if category == ToolCategory.EXEC and args:
        cmd = str(args.get("command") or args.get("cmd") or "").strip().lower()
        if any(pat in cmd for pat in TEST_CMD_PATTERNS):
            return ToolCategory.TEST
        if any(op in cmd for op in BASH_MUTATION_OPS):
            return ToolCategory.EDIT

    if category == ToolCategory.UNKNOWN:
        # Fallback heuristic
        if any(w in clean for w in ("edit", "write", "patch", "replace")):
            return ToolCategory.EDIT
        if any(w in clean for w in ("test", "check", "verify")):
            return ToolCategory.TEST
        if any(w in clean for w in ("bash", "exec", "shell", "run")):
            return ToolCategory.EXEC
        if any(w in clean for w in ("read", "view", "show", "cat")):
            return ToolCategory.READ
        if any(w in clean for w in ("search", "find", "grep", "list")):
            return ToolCategory.SEARCH
        if any(w in clean for w in ("submit", "finish", "complete")):
            return ToolCategory.SUBMIT

    return category


def is_mutation_tool(name: str, args: dict[str, Any] | None = None) -> bool:
    """Returns True if the tool invocation modifies files on disk."""
    return categorize_tool(name, args) == ToolCategory.EDIT


def is_execution_tool(name: str) -> bool:
    """Returns True if the tool executes system commands."""
    return categorize_tool(name) in (ToolCategory.EXEC, ToolCategory.TEST)


def is_verification_tool(
    name: str, args: dict[str, Any] | None = None
) -> bool:
    """Returns True if the tool runs a test suite or verification check."""
    cat = categorize_tool(name, args)
    if cat == ToolCategory.TEST:
        return True
    if clean_name := normalize_tool_name(name):
        if clean_name in ("l0_verify_syntax", "l0_verify_edit_scope", "l0_run_test"):
            return True
    return False


def is_read_tool(name: str) -> bool:
    """Returns True if the tool is an inspection or reading tool."""
    return categorize_tool(name) == ToolCategory.READ


def is_search_tool(name: str) -> bool:
    """Returns True if the tool searches the codebase or directory tree."""
    return categorize_tool(name) == ToolCategory.SEARCH


def is_submit_tool(name: str) -> bool:
    """Returns True if the tool submits or concludes the task."""
    return categorize_tool(name) == ToolCategory.SUBMIT
