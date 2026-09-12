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
