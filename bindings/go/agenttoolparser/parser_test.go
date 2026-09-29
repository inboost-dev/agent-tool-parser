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

func TestCanonicalJSON(t *testing.T) {
	raw := `{"z": 10, "a": {"b": 2, "a": 1}}`
	canon := CanonicalizeJSON(raw)
	expected := `{"a":{"a":1,"b":2},"z":10}`
	if canon != expected {
		t.Fatalf("expected '%s', got '%s'", expected, canon)
	}

	nonJSON := "plain text"
	if CanonicalizeJSON(nonJSON) != "plain text" {
		t.Fatalf("expected nonJSON to be preserved, got '%s'", CanonicalizeJSON(nonJSON))
	}
}

func TestToolCallToCanonicalAndOpenAI(t *testing.T) {
	tc := ToolCall{
		Name: "test_func",
		Args: map[string]interface{}{
			"z": 100,
			"a": "first",
		},
	}

	canonJSON := tc.ToCanonicalJSON()
	expectedArgs := `{"a":"first","z":100}`
	if canonJSON != expectedArgs {
		t.Fatalf("expected '%s', got '%s'", expectedArgs, canonJSON)
	}

	oai := tc.ToOpenAIToolCall("call_custom_123", true)
	if oai.ID != "call_custom_123" {
		t.Fatalf("expected call_custom_123, got %s", oai.ID)
	}
	if oai.Type != "function" {
		t.Fatalf("expected type function, got %s", oai.Type)
	}
	if oai.Function.Name != "test_func" {
		t.Fatalf("expected function name test_func, got %s", oai.Function.Name)
	}
	if oai.Function.Arguments != expectedArgs {
		t.Fatalf("expected canonical arguments '%s', got '%s'", expectedArgs, oai.Function.Arguments)
	}
}

func TestCanonicalizeToolCalls(t *testing.T) {
	calls := []OpenAIToolCall{
		{
			ID:   "call_2",
			Type: "function",
			Function: OpenAIFunction{
				Name:      "write_file",
				Arguments: `{"path":"b.py","content":"bar"}`,
			},
		},
		{
			ID:   "call_1",
			Type: "function",
			Function: OpenAIFunction{
				Name:      "edit_file",
				Arguments: `{"z":1,"a":2}`,
			},
		},
	}

	aligned := CanonicalizeToolCalls(calls)
	if len(aligned) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(aligned))
	}
	// edit_file should sort before write_file
	if aligned[0].Function.Name != "edit_file" {
		t.Fatalf("expected first tool to be edit_file, got %s", aligned[0].Function.Name)
	}
	if aligned[0].Function.Arguments != `{"a":2,"z":1}` {
		t.Fatalf("expected sorted args, got %s", aligned[0].Function.Arguments)
	}
	if aligned[1].Function.Name != "write_file" {
		t.Fatalf("expected second tool to be write_file, got %s", aligned[1].Function.Name)
	}
}

func TestExtractRawToolCalls(t *testing.T) {
	text := `<tool_call>
{"name": "fetch", "arguments": {"city": "Berlin"}}
</tool_call>`
	raws, err := ExtractRawToolCalls(text)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(raws) != 1 {
		t.Fatalf("expected 1 raw call, got %d", len(raws))
	}
	if raws[0].Name != "fetch" {
		t.Fatalf("expected name 'fetch', got %s", raws[0].Name)
	}
	args, err := raws[0].ParseArgs()
	if err != nil {
		t.Fatalf("failed to parse args: %v", err)
	}
	if args["city"] != "Berlin" {
		t.Fatalf("expected city 'Berlin', got %v", args["city"])
	}
}

func TestRepairUnescapedQuotes(t *testing.T) {
	raw := `{"cmd": "echo "hello" >> log.txt"}`
	repaired := RepairUnescapedQuotes(raw)
	expected := `{"cmd": "echo \"hello\" >> log.txt"}`
	if repaired != expected {
		t.Fatalf("expected '%s', got '%s'", expected, repaired)
	}
}

func TestDetectVectorEngine(t *testing.T) {
	engine := DetectVectorEngine()
	if engine == "" {
		t.Fatalf("expected non-empty vector engine name")
	}
}

