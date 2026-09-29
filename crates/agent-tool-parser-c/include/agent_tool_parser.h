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

typedef struct ATPRawToolCall {
    char* name;
    char* raw_args;
    char* raw_source;
    char* call_id;
} ATPRawToolCall;

typedef struct ATPRawToolCallList {
    ATPRawToolCall* calls;
    size_t count;
} ATPRawToolCallList;

/**
 * Extracts raw tool calls from text without full JSON deserialization overhead.
 * Returns an ATPRawToolCallList. The list must be freed with atp_free_raw_tool_call_list().
 */
ATPRawToolCallList atp_extract_raw_tool_calls(const char* text);

/**
 * Parses arguments of a raw tool call into canonical JSON string.
 * Returns allocated string that must be freed with atp_free_string(), or NULL on failure.
 */
char* atp_raw_parse_args(const ATPRawToolCall* call);

/**
 * Frees an ATPRawToolCall allocated by the library.
 */
void atp_free_raw_tool_call(ATPRawToolCall* call);

/**
 * Frees an ATPRawToolCallList allocated by atp_extract_raw_tool_calls().
 */
void atp_free_raw_tool_call_list(ATPRawToolCallList list);

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
 * Normalizes a JSON string into deterministic canonical format with sorted keys.
 * Returns allocated string that must be freed with atp_free_string(), or NULL on failure.
 */
char* atp_canonicalize_json(const char* json_str);

/**
 * Repairs unescaped double quotes inside JSON string literals.
 * Returns allocated string that must be freed with atp_free_string(), or NULL on failure.
 */
char* atp_repair_unescaped_quotes(const char* json_str);

/**
 * Returns static string representing active CPU vector engine (e.g. "avx2_256", "neon_128", "scalar_fallback").
 */
const char* atp_detect_vector_engine(void);

/**
 * Frees any string allocated by this library.
 */
void atp_free_string(char* s);


#ifdef __cplusplus
}
#endif

#endif /* AGENT_TOOL_PARSER_H */
