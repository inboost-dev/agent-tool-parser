// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

package agenttoolparser

import (
	"testing"
)

func TestParseToolCallDSML(t *testing.T) {
	text := `<｜DSML｜tool_calls>
<｜DSML｜tool name=search>
<｜DSML｜parameter name=pattern string=true>ignore-paths</｜DSML｜parameter>
<｜DSML｜parameter name=path string=true>.</｜DSML｜parameter>
</｜DSML｜invoca`

	call, err := ParseToolCall(text)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if call.Name != "search" {
		t.Fatalf("expected name 'search', got '%s'", call.Name)
	}

	if call.Args["pattern"] != "ignore-paths" {
		t.Fatalf("expected pattern 'ignore-paths', got '%v'", call.Args["pattern"])
	}

	if call.Args["path"] != "." {
		t.Fatalf("expected path '.', got '%v'", call.Args["path"])
	}
}

func TestCleanJSONStr(t *testing.T) {
	malformed := "{'tool': 'bash', 'command': 'ls -la'}"
	cleaned := CleanJSONStr(malformed)
	expected := `{"tool": "bash", "command": "ls -la"}`
	if cleaned != expected {
		t.Fatalf("expected '%s', got '%s'", expected, cleaned)
	}
}
