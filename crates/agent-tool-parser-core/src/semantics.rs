// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Semantic categorization of agent tool operations across LLM frameworks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCategory {
    Edit,
    Exec,
    Read,
    Search,
    Test,
    Submit,
    #[serde(rename = "l0_accel")]
    L0Accel,
    Unknown,
}

impl ToolCategory {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Edit => "edit",
            Self::Exec => "exec",
            Self::Read => "read",
            Self::Search => "search",
            Self::Test => "test",
            Self::Submit => "submit",
            Self::L0Accel => "l0_accel",
            Self::Unknown => "unknown",
        }
    }
}

pub fn resolve_canonical_tool(name: &str) -> Option<(&'static str, ToolCategory)> {
    match name {
        // 1. File Modification / Mutation Tools
        "edit" => Some(("edit", ToolCategory::Edit)),
        "write" => Some(("write", ToolCategory::Edit)),
        "str_replace" => Some(("str_replace", ToolCategory::Edit)),
        "write_file" => Some(("write_file", ToolCategory::Edit)),
        "edit_file" => Some(("edit_file", ToolCategory::Edit)),
        "save_file" => Some(("save_file", ToolCategory::Edit)),
        "replace" => Some(("replace", ToolCategory::Edit)),
        "notebookedit" => Some(("notebookedit", ToolCategory::Edit)),
        "apply_diff" => Some(("apply_diff", ToolCategory::Edit)),
        "create_file" => Some(("write_file", ToolCategory::Edit)),
        "modify_file" => Some(("edit_file", ToolCategory::Edit)),
        "patch_file" => Some(("apply_diff", ToolCategory::Edit)),

        // 2. Command Execution Tools
        "bash" => Some(("bash", ToolCategory::Exec)),
        "execute_bash" => Some(("bash", ToolCategory::Exec)),
        "run_bash" => Some(("bash", ToolCategory::Exec)),
        "terminal" => Some(("bash", ToolCategory::Exec)),
        "cmd" => Some(("bash", ToolCategory::Exec)),
        "execute_command" => Some(("bash", ToolCategory::Exec)),
        "run_shell" => Some(("bash", ToolCategory::Exec)),

        // 3. Read / Inspection Tools
        "read" => Some(("read", ToolCategory::Read)),
        "read_file" => Some(("read", ToolCategory::Read)),
        "view" => Some(("read", ToolCategory::Read)),
        "view_file" => Some(("read", ToolCategory::Read)),
        "cat" => Some(("read", ToolCategory::Read)),
        "head" => Some(("read", ToolCategory::Read)),
        "tail" => Some(("read", ToolCategory::Read)),

        // 4. Search Tools
        "grep" => Some(("grep", ToolCategory::Search)),
        "grep_search" => Some(("grep", ToolCategory::Search)),
        "search" => Some(("search", ToolCategory::Search)),
        "file_search" => Some(("file_search", ToolCategory::Search)),
        "find" => Some(("find", ToolCategory::Search)),
        "glob" => Some(("glob", ToolCategory::Search)),
        "list_dir" => Some(("list_dir", ToolCategory::Search)),
        "list_directory" => Some(("list_dir", ToolCategory::Search)),
        "ls" => Some(("list_dir", ToolCategory::Search)),

        // 5. Verification / Testing Tools
        "pytest" => Some(("pytest", ToolCategory::Test)),
        "run_tests" => Some(("run_tests", ToolCategory::Test)),
        "test" => Some(("test", ToolCategory::Test)),
        "l0_run_test" => Some(("l0_run_test", ToolCategory::Test)),

        // 6. Conclude / Submit Tools
        "submit" => Some(("submit", ToolCategory::Submit)),
        "finish" => Some(("finish", ToolCategory::Submit)),
        "complete" => Some(("complete", ToolCategory::Submit)),
        "exit" => Some(("exit", ToolCategory::Submit)),

        // 7. Local L0 Acceleration Tools
        "l0_verify_syntax" => Some(("l0_verify_syntax", ToolCategory::L0Accel)),
        "l0_verify_edit_scope" => Some(("l0_verify_edit_scope", ToolCategory::L0Accel)),
        "l0_clean_test_log" => Some(("l0_clean_test_log", ToolCategory::L0Accel)),
        "l0_view_outline" => Some(("l0_view_outline", ToolCategory::L0Accel)),
        "l0_format_diff" => Some(("l0_format_diff", ToolCategory::L0Accel)),

        _ => None,
    }
}

pub fn resolve_arg_alias(key: &str) -> &'static str {
    match key {
        "file_path" | "filepath" | "filename" | "target_file" | "file" => "path",
        "old_string" | "old_text" | "old_code" => "old_str",
        "new_string" | "new_text" | "new_code" => "new_str",
        "cmd" | "shell_command" => "command",
        "query_string" => "query",
        _ => "",
    }
}

pub const TEST_CMD_PATTERNS: &[&str] = &[
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
];

pub const BASH_MUTATION_OPS: &[&str] = &[
    ">",
    ">>",
    "tee ",
    "patch",
    "sed -i",
    "git apply",
    "git checkout",
];

/// Normalize tool name by lowercasing and resolving canonical mappings.
pub fn normalize_tool_name(raw_name: &str) -> String {
    let clean = raw_name.trim().to_lowercase();
    if clean.is_empty() {
        return String::new();
    }
    if let Some((canon, _)) = resolve_canonical_tool(&clean) {
        canon.to_string()
    } else {
        clean
    }
}

/// Normalize argument keys across agent frameworks (e.g. file_path -> path).
pub fn normalize_tool_args(args: &Value) -> Value {
    match args {
        Value::Object(map) => {
            let mut normalized = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                let target_key = resolve_arg_alias(k);
                let final_key = if target_key.is_empty() {
                    k.as_str()
                } else {
                    target_key
                };
                normalized.insert(final_key.to_string(), v.clone());
            }
            Value::Object(normalized)
        }
        _ => args.clone(),
    }
}

/// Determine semantic category of a tool invocation.
pub fn categorize_tool(name: &str, args: Option<&Value>) -> ToolCategory {
    let clean = name.trim().to_lowercase();
    if clean.is_empty() {
        return ToolCategory::Unknown;
    }

    let category = resolve_canonical_tool(&clean)
        .map(|(_, cat)| cat)
        .unwrap_or(ToolCategory::Unknown);

    if category == ToolCategory::Exec {
        if let Some(args_val) = args {
            let cmd_str = match args_val {
                Value::Object(map) => map
                    .get("command")
                    .or_else(|| map.get("cmd"))
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
                _ => "",
            };
            let cmd_lower = cmd_str.trim().to_lowercase();
            if !cmd_lower.is_empty() {
                if TEST_CMD_PATTERNS.iter().any(|pat| cmd_lower.contains(pat)) {
                    return ToolCategory::Test;
                }
                if BASH_MUTATION_OPS.iter().any(|op| cmd_lower.contains(op)) {
                    return ToolCategory::Edit;
                }
            }
        }
    }

    if category == ToolCategory::Unknown {
        if ["edit", "write", "patch", "replace"]
            .iter()
            .any(|w| clean.contains(w))
        {
            return ToolCategory::Edit;
        }
        if ["test", "check", "verify"]
            .iter()
            .any(|w| clean.contains(w))
        {
            return ToolCategory::Test;
        }
        if ["bash", "exec", "shell", "run"]
            .iter()
            .any(|w| clean.contains(w))
        {
            return ToolCategory::Exec;
        }
        if ["read", "view", "show", "cat"]
            .iter()
            .any(|w| clean.contains(w))
        {
            return ToolCategory::Read;
        }
        if ["search", "find", "grep", "list"]
            .iter()
            .any(|w| clean.contains(w))
        {
            return ToolCategory::Search;
        }
        if ["submit", "finish", "complete"]
            .iter()
            .any(|w| clean.contains(w))
        {
            return ToolCategory::Submit;
        }
    }

    category
}

/// Returns true if the tool invocation modifies files on disk.
pub fn is_mutation_tool(name: &str, args: Option<&Value>) -> bool {
    categorize_tool(name, args) == ToolCategory::Edit
}

/// Returns true if the tool executes system commands.
pub fn is_execution_tool(name: &str) -> bool {
    matches!(
        categorize_tool(name, None),
        ToolCategory::Exec | ToolCategory::Test
    )
}

/// Returns true if the tool runs a test suite or verification check.
pub fn is_verification_tool(name: &str, args: Option<&Value>) -> bool {
    let cat = categorize_tool(name, args);
    if cat == ToolCategory::Test {
        return true;
    }
    let clean = normalize_tool_name(name);
    matches!(
        clean.as_str(),
        "l0_verify_syntax" | "l0_verify_edit_scope" | "l0_run_test"
    )
}

/// Returns true if the tool is an inspection or reading tool.
pub fn is_read_tool(name: &str) -> bool {
    categorize_tool(name, None) == ToolCategory::Read
}

/// Returns true if the tool searches the codebase or directory tree.
pub fn is_search_tool(name: &str) -> bool {
    categorize_tool(name, None) == ToolCategory::Search
}

/// Returns true if the tool submits or concludes the task.
pub fn is_submit_tool(name: &str) -> bool {
    categorize_tool(name, None) == ToolCategory::Submit
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_categorize_tool_mutation() {
        for name in [
            "Edit",
            "edit",
            "write",
            "str_replace",
            "write_file",
            "edit_file",
            "notebookedit",
            "apply_diff",
        ] {
            assert_eq!(categorize_tool(name, None), ToolCategory::Edit);
            assert!(is_mutation_tool(name, None));
            assert!(!is_execution_tool(name));
        }
    }

    #[test]
    fn test_categorize_tool_execution() {
        assert_eq!(categorize_tool("bash", None), ToolCategory::Exec);
        assert_eq!(categorize_tool("execute_bash", None), ToolCategory::Exec);
        assert!(is_execution_tool("bash"));
        assert!(!is_mutation_tool("bash", None));
    }

    #[test]
    fn test_categorize_bash_with_subcommands() {
        let pytest_args = json!({"command": "pytest tests/test_core.py"});
        assert_eq!(
            categorize_tool("bash", Some(&pytest_args)),
            ToolCategory::Test
        );
        assert!(is_verification_tool("bash", Some(&pytest_args)));
        assert!(!is_mutation_tool("bash", Some(&pytest_args)));

        let patch_args = json!({"command": "git apply patch.diff"});
        assert_eq!(
            categorize_tool("bash", Some(&patch_args)),
            ToolCategory::Edit
        );
        assert!(is_mutation_tool("bash", Some(&patch_args)));

        let redirect_args = json!({"command": "echo 'fix' > file.py"});
        assert_eq!(
            categorize_tool("bash", Some(&redirect_args)),
            ToolCategory::Edit
        );
        assert!(is_mutation_tool("bash", Some(&redirect_args)));
    }

    #[test]
    fn test_normalize_tool_name_and_args() {
        assert_eq!(normalize_tool_name("Edit"), "edit");
        assert_eq!(normalize_tool_name("  EXECUTE_BASH  "), "bash");

        let raw = json!({
            "file_path": "src/main.py",
            "old_string": "def old():",
            "new_string": "def new():",
            "cmd": "ls -la"
        });
        let normalized = normalize_tool_args(&raw);
        assert_eq!(normalized["path"], "src/main.py");
        assert_eq!(normalized["old_str"], "def old():");
        assert_eq!(normalized["new_str"], "def new():");
        assert_eq!(normalized["command"], "ls -la");
    }
}
