// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

package agenttoolparser

/*
#cgo CFLAGS: -I${SRCDIR}/../../../crates/agent-tool-parser-c/include
#cgo LDFLAGS: -L${SRCDIR}/../../../target/release -lagent_tool_parser_c -lpthread -ldl -lm
#include "agent_tool_parser.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"errors"
	"sort"
	"strings"
	"unsafe"
)

// ToolCall represents a parsed agent tool invocation.
type ToolCall struct {
	Name      string                 `json:"name"`
	Args      map[string]interface{} `json:"args"`
	RawSource string                 `json:"raw_source"`
}

// ParseToolCall parses a single tool call from text.
func ParseToolCall(text string) (*ToolCall, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	rawCall := C.atp_parse_tool_call(cText)
	if rawCall == nil {
		return nil, errors.New("no valid tool call found")
	}
	defer C.atp_free_tool_call(rawCall)

	tc := &ToolCall{
		Name:      C.GoString(rawCall.name),
		RawSource: C.GoString(rawCall.raw_source),
	}

	argsJSON := C.GoString(rawCall.args_json)
	if argsJSON != "" {
		if err := json.Unmarshal([]byte(argsJSON), &tc.Args); err != nil {
			tc.Args = make(map[string]interface{})
		}
	} else {
		tc.Args = make(map[string]interface{})
	}

	return tc, nil
}

// ParseToolCalls parses all tool calls from text.
func ParseToolCalls(text string) ([]ToolCall, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	list := C.atp_parse_tool_calls(cText)
	defer C.atp_free_tool_call_list(list)

	if list.count == 0 || list.calls == nil {
		return nil, errors.New("no valid tool calls found")
	}

	count := int(list.count)
	slice := unsafe.Slice(list.calls, count)
	results := make([]ToolCall, count)

	for i, c := range slice {
		tc := ToolCall{
			Name:      C.GoString(c.name),
			RawSource: C.GoString(c.raw_source),
		}
		argsJSON := C.GoString(c.args_json)
		if argsJSON != "" {
			if err := json.Unmarshal([]byte(argsJSON), &tc.Args); err != nil {
				tc.Args = make(map[string]interface{})
			}
		} else {
			tc.Args = make(map[string]interface{})
		}
		results[i] = tc
	}

	return results, nil
}

// CleanJSONStr repairs malformed JSON strings.
func CleanJSONStr(raw string) string {
	cStr := C.CString(raw)
	defer C.free(unsafe.Pointer(cStr))

	cleaned := C.atp_clean_json_str(cStr)
	if cleaned == nil {
		return raw
	}
	defer C.atp_free_string(cleaned)

	return C.GoString(cleaned)
}

// OpenAIFunction represents the function definition inside an OpenAI-style tool call.
type OpenAIFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// OpenAIToolCall represents a standard OpenAI function tool call.
type OpenAIToolCall struct {
	ID       string         `json:"id"`
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

// CanonicalSortKeys recursively visits and sorts nested data structures.
// Ensures that nested maps and collections inside interface{} are consistently handled.
func CanonicalSortKeys(v interface{}) interface{} {
	switch val := v.(type) {
	case map[string]interface{}:
		sorted := make(map[string]interface{}, len(val))
		for k, item := range val {
			sorted[k] = CanonicalSortKeys(item)
		}
		return sorted
	case []interface{}:
		sorted := make([]interface{}, len(val))
		for i, item := range val {
			sorted[i] = CanonicalSortKeys(item)
		}
		return sorted
	default:
		return val
	}
}

// CanonicalJSONDumps serializes an object into compact, deterministic JSON with sorted keys.
// Go's encoding/json sorts map keys lexicographically by default.
func CanonicalJSONDumps(v interface{}) (string, error) {
	sorted := CanonicalSortKeys(v)
	b, err := json.Marshal(sorted)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// CanonicalizeJSON parses a JSON string, sorts keys recursively, and re-serializes compactly.
func CanonicalizeJSON(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if !((strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}")) ||
		(strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]"))) {
		return trimmed
	}
	var parsed interface{}
	if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
		return trimmed
	}
	canon, err := CanonicalJSONDumps(parsed)
	if err != nil {
		return trimmed
	}
	return canon
}

// ToCanonicalJSON serializes tool call arguments into deterministic, sorted-key JSON string.
func (tc *ToolCall) ToCanonicalJSON() string {
	if tc.Args == nil {
		return "{}"
	}
	s, err := CanonicalJSONDumps(tc.Args)
	if err != nil {
		return "{}"
	}
	return s
}

// ToOpenAIToolCall converts ToolCall into standard OpenAI tool_call schema object.
func (tc *ToolCall) ToOpenAIToolCall(callID string, canonical bool) OpenAIToolCall {
	id := callID
	if id == "" {
		id = "call_" + tc.Name
	}
	var argsStr string
	if canonical {
		argsStr = tc.ToCanonicalJSON()
	} else {
		b, err := json.Marshal(tc.Args)
		if err != nil {
			argsStr = "{}"
		} else {
			argsStr = string(b)
		}
	}
	return OpenAIToolCall{
		ID:   id,
		Type: "function",
		Function: OpenAIFunction{
			Name:      tc.Name,
			Arguments: argsStr,
		},
	}
}

// CanonicalizeToolCalls normalizes a slice of OpenAI tool calls:
// 1. Ensures arguments are canonicalized JSON.
// 2. Stably sorts tool calls by function name, then by call ID.
func CanonicalizeToolCalls(calls []OpenAIToolCall) []OpenAIToolCall {
	if len(calls) == 0 {
		return calls
	}
	result := make([]OpenAIToolCall, len(calls))
	for i, c := range calls {
		result[i] = OpenAIToolCall{
			ID:   c.ID,
			Type: c.Type,
			Function: OpenAIFunction{
				Name:      c.Function.Name,
				Arguments: CanonicalizeJSON(c.Function.Arguments),
			},
		}
	}

	sort.SliceStable(result, func(i, j int) bool {
		if result[i].Function.Name != result[j].Function.Name {
			return result[i].Function.Name < result[j].Function.Name
		}
		return result[i].ID < result[j].ID
	})

	return result
}
