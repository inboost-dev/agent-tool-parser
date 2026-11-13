# agent-tool-parser ⚡

[![Documentation](https://img.shields.io/badge/docs-interactive%20playground-blueviolet.svg)](https://inboost-dev.github.io/agent-tool-parser/)
[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/agent-tool-parser/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/)
[![Crates.io](https://img.shields.io/badge/crates.io-v0.2.0-orange.svg)](https://crates.io/crates/agent-tool-parser-core)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-red.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)]()
[![Speed](https://img.shields.io/badge/latency-%3C0.05ms-orange.svg)]()

**High-performance tool-call extraction, schema sanitization, and cross-provider bridge for NI/AI.**  
Dual-engine architecture (compiled Rust core with PyO3 acceleration + pure-Python fallback). **Zero external runtime dependencies.** Sub-0.05 millisecond latency.

---

## Quickstart (30 Seconds)

### Installation

```bash
# Pure-Python standard install (zero external dependencies)
pip install agent-tool-parser

# Optional extras:
pip install "agent-tool-parser[all]"      # installs orjson and json-repair
pip install "agent-tool-parser[fast]"     # installs orjson for accelerated pure-Python JSON
```

### 3-Line Example

```python
from agent_tool_parser import parse_tool_call

# Handles reasoning tokens, DSML markup, XML tags, and unquoted parameters automatically:
raw_output = """
<think>Checking test suite before creating commit...</think>
<｜tool call begin｜>function=bash<｜tool sep｜>{"command": "pytest tests/"}<｜tool call end｜>
"""

call = parse_tool_call(raw_output)
print(call.name)            # 'bash'
print(call.args)            # {'command': 'pytest tests/'}
print(call.is_verification) # True (automatically identifies test and verification commands!)
```

---

## Cheatsheet for NI/AI

Fast lookup table for NI/AI (Claude Code, Cursor, Copilot, Antigravity):

| Goal / Task | Function / Class | Input Type | Output Type | Latency |
| :--- | :--- | :--- | :--- | :---: |
| **Extract 1 tool call** | `parse_tool_call(text)` | `str` | `ToolCall(name, args)` | < 0.05 ms |
| **Extract all calls (safe)** | `try_parse_tool_calls(text)` | `str` | `list[ToolCall]` (empty if none) | < 0.05 ms |
| **Stream token-by-token** | `StreamingToolParser().feed(chunk)` | `str` | `list[StreamEvent]` | < 0.005 ms |
| **Fix Gemini 400 Bad Request** | `sanitize_tool_schema_for_gemini(schema)` | `dict` (JSON Schema) | `dict` (Gemini-compliant) | < 0.01 ms |
| **OpenAI Strict Mode** | `sanitize_tool_schema_for_openai(schema, strict=True)` | `dict` | `dict` (additionalProperties=False) | < 0.01 ms |
| **Translate Request** | `translate_request(req, from_fmt, to_fmt)` | `dict` | `dict` (OpenAI ↔ Claude ↔ Gemini) | < 0.02 ms |
| **Translate Response** | `translate_response(resp, from_fmt, to_fmt)` | `dict` | `dict` | < 0.01 ms |
| **Translate SSE Stream Line** | `create_streaming_transformer(...).transform_sse_line(line)` | `str` (SSE line) | `list[str]` (target SSE lines) | < 0.005 ms |
| **Bridge to MCP Tool Call** | `to_mcp_tool_call(output)` | `str \| dict` | `{"jsonrpc": "2.0", "method": "tools/call", ...}` | < 0.01 ms |
| **Convert Tool Result to MCP** | `to_mcp_call_result(content, is_error=False)` | `Any` | `{"content": [...], "isError": bool}` | < 0.005 ms |
| **A2A Audit & Metadata Headers** | `ToolMetadata(...).to_headers()` | `ToolMetadata` | `dict[str, str]` (`x-a2a-*`) | < 0.005 ms |
| **Preserve KV-Cache** | `canonicalize_tool_calls(tool_calls)` | `list[dict]` | `list[dict]` (sorted keys & items) | < 0.01 ms |
| **Repair Broken JSON** | `safe_json_loads(corrupt_str)` | `str` | `dict \| list` | < 0.01 ms |

---

## Core Capabilities & Practical Recipes

### 1. Extract & Repair Tool Calls from Any Model
Open-weights models (DeepSeek R1/V3, Qwen 2.5, Llama 3) and API aggregators often output mixed text containing chain-of-thought (`<think>...</think>`), XML tags, AST function calls, or unescaped quotes in shell scripts. `agent-tool-parser` extracts clean structured data deterministically:

```python
from agent_tool_parser import try_parse_tool_calls

raw_output = """
Evaluating file modifications.
<tool_use>
  <name>edit_file</name>
  <arguments>{"path": "src/auth.py", "patch": "def verify():\n    return True\n"}</arguments>
</tool_use>
"""

calls = try_parse_tool_calls(raw_output)
for call in calls:
    print(f"Tool: {call.name}")
    print(f"Args: {call.args}")
```

### 2. Token-by-Token Event Streaming (SSE & Real-Time Sandboxing)
Feed tokens one by one as they arrive from an inference stream. Receive instant events for text, thoughts, early tool starts, argument deltas, and completed calls:

```python
from agent_tool_parser import StreamingToolParser

parser = StreamingToolParser()

async for chunk in model_token_stream:
    for event in parser.feed(chunk.text):
        if event.is_text:
            await stream_to_client(event.text)
        elif event.is_thinking:
            await stream_thoughts_to_ui(event.thinking)
        elif event.is_tool_call_started:
            # Low latency win: warm up sandbox or request NI confirmation
            # before argument generation finishes!
            await prepare_sandbox(event.tool_name)
        elif event.is_tool_call_arguments_chunk:
            await stream_argument_delta(event.delta)
        elif event.is_tool_call_completed:
            await dispatch_tool_execution(event.tool_call)

# Recover any truncated calls if generation cut off mid-stream:
for event in parser.finish():
    if event.is_tool_call_completed:
        await dispatch_tool_execution(event.tool_call)
```

### 3. Universal Provider Translation (OpenAI ↔ Anthropic ↔ Gemini)
Zero-dependency $O(N)$ canonical translation for requests, responses, and real-time SSE lines. Ideal for API gateways and reverse proxies (e.g. `inboost-ai-proxy`, LiteLLM, FastAPI middleware):

```python
from agent_tool_parser import (
    translate_request,
    translate_response,
    create_streaming_transformer,
    ProtocolRegistry,
)

# 1. Translate Request: OpenAI format -> Google Gemini format
gemini_req = translate_request(
    openai_request,
    from_format="openai",
    to_format="gemini",
    override_model="gemini-2.5-pro",
)

# 2. Translate Response: Gemini format -> OpenAI ChatCompletion format
openai_resp = translate_response(
    gemini_resp,
    from_format="gemini",
    to_format="openai",
    model="gpt-4o",
)

# 3. Stream SSE lines on the fly (300,000 chunks/sec):
transformer = create_streaming_transformer(from_format="gemini", to_format="openai", model="gpt-4o")
for output_sse_line in transformer.transform_sse_line(incoming_gemini_sse_line):
    yield output_sse_line

# 4. Use built-in provider presets (copilot, cursor, opencode, deepseek, mistral, azure):
custom_req = translate_request(req, from_format="claude", to_format="copilot")
```

### 4. Sanitize Tool Schemas (Fix Gemini HTTP 400 & OpenAI Strict)
Standard JSON Schemas from Pydantic, OpenAPI, or MCP routinely crash Google Gemini with `400 Bad Request` because Gemini rejects `$schema`, `$defs`, `$ref`, `default`, `examples`, and `anyOf`. `agent-tool-parser` provides deterministic schema sanitization:

```python
from agent_tool_parser import sanitize_tool_schema_for_gemini, sanitize_tool_schema_for_openai

# Resolves $defs/$ref, flattens anyOf, and strips unsupported keys for Gemini:
safe_gemini_schema = sanitize_tool_schema_for_gemini(raw_pydantic_schema)

# Enforces additionalProperties=False and requires all fields for OpenAI Strict mode:
safe_openai_schema = sanitize_tool_schema_for_openai(raw_schema, strict=True)
```

### 5. Model Context Protocol (MCP) & Agent-to-Agent (A2A) Governance
Directly bridge model outputs to Anthropic's MCP JSON-RPC specification, convert tool definitions, and attach A2A audit headers across multi-agent handoffs:

```python
from agent_tool_parser import (
    to_mcp_tool_call,
    to_mcp_call_result,
    convert_tool_definitions,
    ToolMetadata,
)

# 1. Raw text or OpenAI dict -> MCP JSON-RPC 2.0 tools/call
mcp_rpc = to_mcp_tool_call('<tool_call>{"name": "read_file", "arguments": {"path": "main.py"}}</tool_call>')
# -> {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "main.py"}}}

# 2. Result -> standard MCP CallToolResult:
mcp_res = to_mcp_call_result({"status": "saved", "bytes": 1024})

# 3. MCP tools/list -> Gemini functionDeclarations (with auto-sanitization):
gemini_tools = convert_tool_definitions(mcp_tools, from_format="mcp", to_format="gemini")

# 4. A2A Governance & Provenance Headers (audit, approvals, idempotency):
meta = ToolMetadata(
    caller_agent="agent-supervisor",
    requires_approval=True,
    is_mutation=True,
    security_level="restricted",
    custom_annotations={"tenant": "prod-eu"},
)
headers = meta.to_headers(prefix="x-a2a-")
# -> {"x-a2a-caller-agent": "agent-supervisor", "x-a2a-requires-approval": "true", ...}
```

### 6. KV-Cache Invariance & Semantic Inspection
In multi-turn agent loops on vLLM or SGLang, permuted JSON argument keys invalidate prefix KV-caches. Canonicalize tool calls before sending them back to inference, and inspect tool actions for safety policies:

```python
from agent_tool_parser import canonicalize_tool_calls, parse_tool_call, ToolCategory

# 1. Deterministic sorting for KV-cache reuse:
messages[-1]["tool_calls"] = canonicalize_tool_calls(messages[-1]["tool_calls"])

# 2. Semantic inspection of actions:
call = parse_tool_call(model_output)
if call.is_mutation:
    require_ni_confirmation(call)
if call.is_verification:
    log_test_run(call)
print(call.category)  # ToolCategory.EDIT, ToolCategory.TEST, ToolCategory.EXEC, etc.
```

---

## Multi-Language Quickstart

### Go (`cgo`)

```go
package main

import (
    "fmt"
    "log"
    agenttoolparser "github.com/inboost-dev/agent-tool-parser/bindings/go/agenttoolparser"
)

func main() {
    raw := `<｜tool calls begin｜><｜tool call begin｜>function=search<｜tool sep｜>{"pattern": "parse", "path": "."}<｜tool call end｜><｜tool calls end｜>`
    call, err := agenttoolparser.ParseToolCall(raw)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Tool: %s, Args: %v\n", call.Name, call.Args)
}
```

### Rust (Native Crate)

```rust
use agent_tool_parser_core::parse_tool_call;

fn main() {
    let raw = "<invoke name=\"bash\"><parameter name=\"cmd\">git status</parameter></invoke>";
    let call = parse_tool_call(raw).expect("Valid tool invocation");
    println!("Tool: {}, Args: {:?}", call.name, call.args);
}
```

### C / C++ (`agent_tool_parser.h`)

```c
#include "agent_tool_parser.h"
#include <stdio.h>

int main() {
    const char* raw = "<invoke name=\"bash\"><parameter name=\"cmd\">pwd</parameter></invoke>";
    ATPToolCall* call = atp_parse_tool_call(raw);
    if (call) {
        printf("Tool: %s\nArguments: %s\n", call->name, call->args_json);
        atp_free_tool_call(call);
    }
    return 0;
}
```

---

## Supported Formats

| Format Specification | Representative Input | Extraction Status |
| :--- | :--- | :---: |
| **Anthropic Claude XML** | `<tool_use><name>read_file</name><arguments>{"path": "src/core.py"}</arguments></tool_use>` | Supported |
| **OpenAI Function Format** | `{"tool_calls": [{"type": "function", "function": {"name": "search", "arguments": "{...}"}}]}` | Supported |
| **Qwen / Open-Weights XML** | `<function=search>\n<parameter=query>vector</parameter>\n</function>` | Supported |
| **Container Tag Wrappers** | `<tool_call><function=search>...</function></tool_call>` | Supported |
| **DeepSeek Native Tokens** | `<｜tool call begin｜>function=bash<｜tool sep｜>{"command": "pytest"}<｜tool call end｜>` | Supported |
| **DeepSeek DSML** | `<｜DSML｜tool name=search><｜DSML｜parameter name=query>vector</｜DSML｜parameter>` | Supported |
| **Qwen-Agent Tokens** | `<\|action_start\|><\|action_name\|>run<\|action_args\|>{"script": "main.py"}<\|action_end\|>` | Supported |
| **Llama 3.1+ Python Tag** | `<\|python_tag\|>bash.call(command="pytest")` | Supported |
| **Mistral Tool Syntax** | `[TOOL_CALLS] [{"name": "search", "arguments": {"query": "vector"}}]` | Supported |
| **Python Expression Syntax** | `read_file(path="src/main.py", offset=10)` | Supported |
| **ReAct Action Notation** | `Action: bash\nAction Input: pytest` | Supported |
| **Hermes 2/3 XML Markup** | `<tool_call>{"name": "search", "arguments": {"query": "vector"}}</tool_call>` | Supported |
| **Direct Parameter XML** | `<tool_name>search</tool_name><query>vector</query>` | Supported |
| **Markdown Fenced JSON** | ````json\n{"tool": "bash", "command": "git status"}\n```` | Supported |
| **Embedded String CDATA** | `<parameter name="patch"><![CDATA[def test():\n    assert True]]></parameter>` | Supported |
| **Malformed JSON Repair** | `{'tool': 'bash', 'command': 'pytest', 'verbose': True,}` | Supported |
| **Thinking Blocks** | `<think>Analyzing context...</think>` | Delimited & Stripped |

---

## Architecture Deep Dive & Trade-offs

### Why Client-Side Parsing?

When extracting structured tool invocations from LLM outputs, teams typically choose between three system boundaries:

| Architectural Dimension | Client-Side Post-Parsing (`agent-tool-parser`) | Logit-Level Constrained Decoding (e.g., XGrammar, Outlines) | Secondary Model Pass (Neural Re-Parser) |
| :--- | :--- | :--- | :--- |
| **Execution Boundary** | Client runtime (post-generation) | Inference engine sampling loop | Secondary inference endpoint or worker |
| **Inference Engine Access** | Engine-agnostic (processes standard HTTP / REST text streams) | Requires direct access to logits and the sampling pipeline | Requires secondary model invocation or separate service deployment |
| **Runtime Dependencies** | Python standard library (zero external dependencies) | Inference runtime stack (e.g., PyTorch, CUDA, Triton) | Model inference framework or API client |
| **Provider API Support** | Compatible with any endpoint returning text (OpenRouter, Groq, Together, etc.) | Incompatible with third-party APIs lacking custom sampler controls | Compatible via sequential API round-trip |
| **Grammar Enforcement Mechanism** | Deterministic lexical parsing and structural repair | Logit masking via finite-state machine (FSM) during token sampling | Probabilistic generation guided by system prompt |
| **Format Scope** | Multi-format: XML, DSML, Markdown JSON, Python AST, CDATA blocks | Strictly defined JSON Schemas or context-free grammars (EBNF) | Unconstrained (dependent on secondary model training distribution) |

### Key Trade-offs

1. **Third-Party Inference Endpoints:** When utilizing managed providers (OpenRouter, DeepSeek API, Groq, Fireworks), clients receive serialized text rather than sampler-level logit access. Server-side grammar constraints are simply unavailable.
2. **Intermediate Reasoning Tokens:** When reasoning models emit unconstrained chain-of-thought blocks (`<think>...</think>`), logit-level grammars must support dynamic transitions between free-form text and structured output schemas. Client-side extraction cleanly separates thoughts from structured payloads.
3. **Payload Heterogeneity:** Tool calls frequently involve non-JSON payloads, such as multi-line shell scripts, unified diffs, or XML CDATA blocks. Client-side parsers process these delimiters without requiring escape sequences.

---

## Dual-Engine Architecture

```
agent-tool-parser/
├── src/agent_tool_parser/          # Pure-Python engine (zero-dependency, no compiler required)
│   ├── _accelerated.so             # Optional PyO3 compiled native extension
│   └── __init__.py                 # Transparent runtime dispatch (ACCELERATED flag)
├── crates/
│   ├── agent-tool-parser-core/     # High-performance Rust core extraction engine
│   ├── agent-tool-parser-c/        # C-ABI export & standard C header (agent_tool_parser.h)
│   ├── agent-tool-parser-py/       # PyO3 bindings with pythonize zero-copy deserialization
│   └── agent-tool-parser-wasm/     # WebAssembly bindings (wasm-bindgen) for Node.js & browser
└── bindings/
    └── go/agenttoolparser/         # Native Go module wrapping C-ABI via cgo
```

- **Parity Guarantee:** Both the pure-Python and Rust core engines pass the exact same validation suite (191 Python unit & parity tests, 102 Rust workspace tests, and 260+ differential parity assertions) covering all supported formats, error handling, parameter aliases, and degenerate/adversarial input texts.
- **Zero-Compile Fallback:** If the compiled native extension is not present (or when `AGENT_TOOL_PARSER_NO_EXT=1` is set in the environment), the package operates in pure Python with zero external dependencies.
- **Compiled Acceleration:** When the compiled extension is available, parsing execution occurs entirely in compiled native code with sub-0.05 ms latency.

---

## API Reference

### Functional Interface
- `parse_tool_call(text: str) -> ToolCall`: Extracts a single tool call from text. Raises `ToolError` if no valid structure is recognized.
- `parse_tool_calls(text: str) -> list[ToolCall]`: Extracts all tool calls present in the text. Raises `ToolError` if none detected.
- `try_parse_tool_call(text: str) -> Optional[ToolCall]`: Non-raising variant. Returns `None` if no invocation is detected.
- `try_parse_tool_calls(text: str) -> list[ToolCall]`: Non-raising variant. Returns empty list `[]` if no invocations detected.
- `safe_json_loads(s: str, default: Any = None) -> Any`: Standalone utility for resilient JSON decoding (resolves trailing commas, single quotes, unclosed brackets).

### Schema Sanitization & MCP
- `sanitize_tool_schema_for_gemini(schema: dict) -> dict`: Inlines `$defs`/`$ref`, flattens `anyOf`, and strips unsupported keys (`$schema`, `default`, `examples`, `additionalProperties`) preventing Google Gemini HTTP 400.
- `sanitize_tool_schema_for_openai(schema: dict, strict: bool = True) -> dict`: Enforces `additionalProperties: False` and adds all properties to `required` for OpenAI Structured Outputs.
- `to_mcp_tool_call(output: str | dict) -> dict`: Extracts tool call and returns standard MCP JSON-RPC 2.0 `tools/call`.
- `to_mcp_call_result(content: Any, is_error: bool = False) -> dict`: Creates standard MCP `CallToolResult`.
- `convert_tool_definitions(tools, from_format, to_format) -> list[dict]`: Bidirectional schema conversion across OpenAI, Anthropic, Gemini, and MCP.
- `convert_tool_calls(calls, from_format, to_format) -> list[dict]`: Converts tool invocations across ecosystems.
- `convert_tool_results(results, from_format, to_format) -> list[dict]`: Converts tool execution outcomes across ecosystems.

### Cross-Provider Translation
- `translate_request(payload, from_format, to_format, override_model=None) -> dict`: Translates requests between OpenAI, Anthropic, Gemini, and registered aliases.
- `translate_response(payload, from_format, to_format, model=None) -> dict`: Translates responses, preserving text, tool calls, thinking blocks, and errors.
- `create_streaming_transformer(from_format, to_format, model=None) -> StreamingProtocolTransformer`: Creates real-time SSE line transformer.

### KV-Cache & Semantic Inspection
- `canonicalize_tool_calls(tool_calls: list[dict]) -> list[dict]`: Deterministically sorts argument keys and tool call arrays for prefix KV-cache reuse.
- `categorize_tool(name: str, args: dict = None) -> ToolCategory`: Classifies into `EDIT`, `EXEC`, `READ`, `SEARCH`, `TEST`, `SUBMIT`, `L0_ACCEL`.
- `ToolCall.is_mutation`: `True` if tool mutates files or code.
- `ToolCall.is_verification`: `True` if tool runs tests or assertion checks.

---

## Performance Characteristics

Measurements conducted on Python 3.12 (AMD EPYC, single execution thread):

| Evaluation Scenario | Average Latency | Throughput |
| :--- | :---: | :---: |
| Request Translation (`translate_request`) | **0.017 ms** | ~58,000 ops / sec |
| Streaming Chunk Translation (`transform_chunk`) | **0.003 ms** | ~300,000 chunks / sec |
| DSML Markup Extraction | **0.038 ms** | ~26,000 calls / sec |
| DeepSeek Delimiter Parsing | **0.024 ms** | ~41,000 calls / sec |
| Markdown & Malformed JSON Repair | **0.015 ms** | ~65,000 calls / sec |

---

## Automated Testing & CI

All pull requests and commits are continuously validated across runtimes in GitHub Actions:
- **Python Multi-Version Matrix:** Python 3.9, 3.10, 3.11, 3.12, 3.13 running `ruff`, `mypy`, and the complete 191-test suite under pure-Python fallback (`AGENT_TOOL_PARSER_NO_EXT=1`).
- **Rust Core & Multi-Language Bindings:** `cargo test --workspace` (102 unit, integration, degenerate, and benchmark tests).
- **Parity Verification:** Dual-engine differential verification (`tests/test_parity.py` with 260+ assertions).

---

## Contributing

We welcome contributions to `agent-tool-parser`. Because the project maintains dual implementations (pure-Python reference and compiled Rust core), all pull requests modifying parsing heuristics, format extractors, or cleaner rules must provide synchronized changes across both engines.

See our [**Contributing Guide (CONTRIBUTING.md)**](CONTRIBUTING.md) for full instructions and checklists.

---

## Security & Execution Disclaimer

`agent-tool-parser` is strictly a **passive lexical and structural parser**. It does not validate, authenticate, sandbox, or execute tool calls, shell commands, or arbitrary parameters emitted by language models. Parsed tool calls must **never** be executed directly without strict downstream input validation, authorization checks, and process isolation.

See our full [**Legal & Execution Disclaimer (DISCLAIMER.md)**](DISCLAIMER.md).

---

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 InBoost Team.
