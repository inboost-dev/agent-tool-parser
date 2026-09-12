// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

#ifndef AGENT_TOOL_PARSER_H
#define AGENT_TOOL_PARSER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ATPToolCall {
    char* name;
    char* args_json;
    char* raw_source;
} ATPToolCall;

typedef struct ATPToolCallList {
    ATPToolCall* calls;
    size_t count;
} ATPToolCallList;

/**
 * Parses a single tool call from text.
 * Returns NULL if no valid tool call could be extracted.
 * The returned pointer must be freed with atp_free_tool_call().
 */
ATPToolCall* atp_parse_tool_call(const char* text);

/**
 * Parses all tool calls from text.
 * Returns an ATPToolCallList. The list must be freed with atp_free_tool_call_list().
 */
ATPToolCallList atp_parse_tool_calls(const char* text);

/**
 * Frees an ATPToolCall allocated by atp_parse_tool_call().
 */
void atp_free_tool_call(ATPToolCall* call);

/**
 * Frees an ATPToolCallList allocated by atp_parse_tool_calls().
 */
void atp_free_tool_call_list(ATPToolCallList list);

/**
 * Repairs and validates a JSON string.
 * Returns allocated string that must be freed with atp_free_string(), or NULL on failure.
 */
char* atp_clean_json_str(const char* json_str);

/**
 * Frees any string allocated by this library.
 */
void atp_free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif /* AGENT_TOOL_PARSER_H */
