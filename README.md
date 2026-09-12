# agent-tool-parser ⚡

[![Documentation](https://img.shields.io/badge/docs-interactive%20playground-blueviolet.svg)](https://inboost-dev.github.io/agent-tool-parser/)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.1-blue.svg)](https://pypi.org/project/agent-tool-parser/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/)
[![Crates.io](https://img.shields.io/badge/crates.io-v0.1.1-orange.svg)](https://crates.io/crates/agent-tool-parser-core)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-red.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)]()
[![Speed](https://img.shields.io/badge/latency-%3C0.05ms-orange.svg)]()

**High-performance, dual-engine (Rust + Python) parser and cleaner for heterogeneous LLM tool calls and structured outputs.**  
[**Interactive Documentation & Live Playground ↗**](https://inboost-dev.github.io/agent-tool-parser/)

`agent-tool-parser` is a high-performance library with a dual-engine architecture (compiled Rust core with PyO3 acceleration + pure-Python universal fallback) designed to extract, normalize, and repair tool calls from large language model outputs. It processes standard OpenAI schemas, Anthropic Claude XML, DeepSeek DSML and native tokens, Qwen tokens, Llama AST syntax, and truncated JSON payloads with sub-0.05 millisecond latency and zero required runtime dependencies. Cross-language bindings are available for Python, Rust, Go, C/C++, and WebAssembly.

---

## Design Rationale & Architectural Context

### Scope and Applicability
When integrating exclusively with closed-source APIs (such as OpenAI or Anthropic) via their native client SDKs, server-side infrastructure typically frames tool calls into structured protocol objects prior to delivery. In such architectures, an external parser is generally unnecessary.

However, in systems integrating open-weights models, local inference engines (e.g., vLLM, SGLang, Ollama), or third-party inference aggregators (e.g., OpenRouter, Together AI, Groq), tool invocations are frequently delivered within the unsegmented text stream (`message.content`). Common integration challenges include:

1. **Delimiter Heterogeneity**: Models output diverse serialization schemes, including XML tags (`<tool_use>`), DSML blocks (`<｜DSML｜invoke>`), native tokens (`<｜tool call begin｜>`), or AST function calls (`call:tool(arg=val)`).
2. **Co-occurring Chain-of-Thought Blocks**: Reasoning models output internal deliberation traces (such as `<think>...</think>`) in the same stream as the final tool invocation.
3. **Code Payload Escaping**: Serializing multi-line scripts, shell commands, or unified diffs inside standard JSON strings frequently leads to escape sequence errors with quotes and whitespace.
4. **Parameter Name Variations**: Minor model deviations in argument naming (e.g., `cmd` instead of `command`, `file` instead of `path`) can cause downstream schema validation failures.
5. **Stream Truncation**: When generation terminates prematurely due to token budget limits, partial JSON structures fail standard deserialization.

`agent-tool-parser` provides a deterministic, zero-dependency processing layer to normalize these outputs before execution.

---

## Comparative Analysis

| Approach | Design Focus | Execution Characteristics | Scope & Constraints |
| :--- | :--- | :--- | :--- |
| **`agent-tool-parser`** | Tool invocation extraction & normalization | Sub-0.05 ms latency; zero dependencies; dual-engine (Rust + Python). | Specialized for tool calling; supports XML, DSML, AST, and JSON repair. |
| **`json-repair`** | General-purpose JSON syntax repair | Lightweight; standard JSON normalization. | Restricted to JSON grammar; does not process markup tags, thinking blocks, or argument schemas. |
| **`LangChain` / `LlamaIndex`** | Full-stack application frameworks | Comprehensive multi-package ecosystem. | Higher memory footprint and dependency tree for pipelines requiring only output parsing. |
| **`Instructor`** | Pydantic validation via function calling | Enforces structured types against server schemas. | Requires provider-level schema support; does not extract delimiter-based markup from message text. |
| **Guided Decoding (`vLLM / SGLang`)** | Constrained token generation | Enforces grammar during sampling. | Can restrict intermediate reasoning steps in CoT models; inapplicable when calling third-party API endpoints. |
| **Ad-Hoc Regular Expressions** | Project-specific pattern matching | Zero external dependencies. | Brittle against unquoted attributes, nested tags, and varying model formatting conventions. |

### Architectural Trade-offs: Post-Parsing vs. Server-Side Alternatives

In production agent architectures, extracting structured tool invocations is typically approached at one of three system boundaries:

| Architectural Dimension | Client-Side Post-Parsing (`agent-tool-parser`) | Logit-Level Constrained Decoding (e.g., XGrammar, Outlines) | Secondary Model Pass (Neural Re-Parser) |
| :--- | :--- | :--- | :--- |
| **Execution Boundary** | Client runtime (post-generation) | Inference engine sampling loop | Secondary inference endpoint or worker |
| **Inference Engine Access** | Engine-agnostic (processes standard HTTP / REST text streams) | Requires direct access to logits and the sampling pipeline | Requires secondary model invocation or separate service deployment |
| **Runtime Dependencies** | Python standard library (zero external dependencies) | Inference runtime stack (e.g., PyTorch, CUDA, Triton) | Model inference framework or API client |
| **Provider API Support** | Compatible with any endpoint returning text (OpenRouter, Groq, Together, etc.) | Incompatible with third-party APIs lacking custom sampler controls | Compatible via sequential API round-trip |
| **Grammar Enforcement Mechanism** | Deterministic lexical parsing and structural repair | Logit masking via finite-state machine (FSM) during token sampling | Probabilistic generation guided by system prompt |
| **Format Scope** | Multi-format: XML, DSML, Markdown JSON, Python AST, CDATA blocks | Strictly defined JSON Schemas or context-free grammars (EBNF) | Unconstrained (dependent on secondary model training distribution) |

#### Architectural Trade-offs

- **API Boundary & Sampler Control:** Logit masking operates inside the autoregressive generation loop on the inference host. When utilizing managed or third-party inference providers (such as OpenRouter, DeepSeek API, or Groq), clients receive serialized text rather than sampler-level logit access, making server-side grammar constraints unavailable.
- **Intermediate Reasoning Tokens:** When reasoning architectures emit unconstrained chain-of-thought blocks (such as `<think>...</think>`), logit-level grammars must support dynamic transitions between free-form text and structured output schemas. Client-side extraction isolates the deliberation trace from the structured invocation payload after generation completes.
- **Payload Heterogeneity:** Tool calls frequently involve non-JSON payloads, such as multi-line shell scripts, unified diffs, or XML CDATA blocks. Client-side parsers process these alternative delimiters directly without requiring JSON string escaping.

---

## Architectural Reference Patterns

### Pattern 1: Autonomous Agent Loop (ReAct / Task Execution)
Standard ReAct loops require resilient extraction when alternating between reasoning and tool execution:

```python
from agent_tool_parser import try_parse_tool_calls


def run_agent_turn(model_output: str) -> list[dict]:
    # Extracts all valid tool calls, removing any <think> blocks
    calls = try_parse_tool_calls(model_output)

    results = []
    for call in calls:
        execution_result = dispatch_tool(call.name, call.args)
        results.append({"tool": call.name, "result": execution_result})

    return results
```

### Pattern 2: API Gateway & Proxy Normalization (LiteLLM / FastAPI Middleware)
When proxying multi-provider models into standard OpenAI-compatible endpoints:

```python
from agent_tool_parser import try_parse_tool_calls


def normalize_completion_response(response_dict: dict) -> dict:
    choice = response_dict["choices"][0]
    message = choice["message"]

    # If the provider omitted tool_calls but returned markup in content:
    if not message.get("tool_calls") and message.get("content"):
        extracted = try_parse_tool_calls(message["content"])
        if extracted:
            message["tool_calls"] = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.args)},
                }
                for i, c in enumerate(extracted)
            ]
    return response_dict
```

### Pattern 3: Control Plane & Fleet Worker Ingress (Mission Control / Paperclip / Dify)
In multi-agent environments managing autonomous background workers, normalizing arguments prior to approval gates prevents stalled task queues:

```python
from agent_tool_parser import ToolParser, ToolError

parser = ToolParser(
    allowed_tools={"read_file", "write_file", "execute_command"},
    param_aliases={"cmd": "command", "filepath": "path"},
)


def process_worker_step(raw_step_output: str) -> dict:
    try:
        call = parser.parse(raw_step_output)
        return {"status": "ready_for_approval", "call": call}
    except ToolError as error:
        return {"status": "rejected", "error": str(error)}
```

### Pattern 4: Asynchronous Parallel Tool Dispatch
Processing multiple tool calls from a single model generation step:

```python
import asyncio
from agent_tool_parser import parse_tool_calls


async def handle_parallel_generation(model_output: str):
    calls = parse_tool_calls(model_output)
    tasks = [async_execute(call.name, call.args) for call in calls]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

---

## Installation

```bash
# Standard Python installation (zero required dependencies, pure Python fallback by default)
pip install agent-tool-parser

# With optional performance and advanced JSON repair extras:
pip install "agent-tool-parser[all]"      # installs orjson and json-repair
pip install "agent-tool-parser[fast]"     # installs orjson for accelerated pure-Python JSON
pip install "agent-tool-parser[repair]"   # installs json-repair for advanced corrupt JSON

# Go module
go get github.com/inboost-dev/agent-tool-parser/bindings/go/agenttoolparser
```

---

## Quickstart

### Python

```python
from agent_tool_parser import parse_tool_call, ACCELERATED

print(f"Accelerated native core active: {ACCELERATED}")

# Example response containing reasoning tokens, DSML markup, and unquoted parameters:
raw_output = """
<think>
Evaluating dependency versions before test execution.
</think>
<｜DSML｜tool_calls>
<｜DSML｜invoke name="run_tests">
<｜DSML｜parameter name="suite">unit</｜DSML｜parameter>
<｜DSML｜parameter name="verbose">true</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
"""

call = parse_tool_call(raw_output)

print(call.name)  # 'run_tests'
print(call.args)  # {'suite': 'unit', 'verbose': True}
print(call.raw_source)  # Raw matched substring
```

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

## Dual-Engine Architecture & Multi-Language Core

`agent-tool-parser` is structured with a modular, dual-engine design:

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

### Engine Dispatch & Synchronization

- **Parity Guarantee**: Both the pure-Python and Rust core engines pass the exact same validation suite (104 Python unit tests, 58 Rust unit/integration/benchmark tests, and 240+ differential parity assertions) covering all supported formats, error handling, parameter aliases, and degenerate/adversarial input texts.
- **Zero-Compile Fallback**: If the compiled native extension is not present (or when `AGENT_TOOL_PARSER_NO_EXT=1` is set in the environment), the package operates in pure Python with zero external dependencies.
- **Compiled Acceleration**: When the compiled extension is available, parsing execution occurs entirely in compiled native code with sub-millisecond execution times.

## Supported Formats

| Format Specification | Representative Input | Extraction Status |
| :--- | :--- | :---: |
| **Anthropic Claude XML** | `<tool_use><name>read_file</name><arguments>{"path": "src/core.py"}</arguments></tool_use>` | Supported |
| **OpenAI Function Format** | `{"tool_calls": [{"type": "function", "function": {"name": "search", "arguments": "{...}"}}]}` | Supported |
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

## API Reference

### Functional Interface

#### `parse_tool_call(text: str, ...) -> ToolCall`
Extracts a single tool call from text. Raises `ToolError` if no valid structure is recognized.

#### `parse_tool_calls(text: str, ...) -> list[ToolCall]`
Extracts all tool calls present in the text. Raises `ToolError` if the output contains no extractable invocations.

#### `try_parse_tool_call(text: str, ...) -> Optional[ToolCall]`
Non-raising variant of `parse_tool_call`. Returns `None` if no invocation is detected.

#### `try_parse_tool_calls(text: str, ...) -> list[ToolCall]`
Non-raising variant of `parse_tool_calls`. Returns an empty list `[]` if no invocations are detected.

#### `safe_json_loads(s: str, default: Any = None) -> Any`
Standalone utility for resilient JSON decoding. Resolves trailing commas, single-quoted keys, unquoted boolean/null literals, and unclosed brackets.

---

### Object-Oriented Interface: `ToolParser`

The `ToolParser` class allows setting persistent validation rules, allowed tool sets, and parameter mappings across calls:

```python
from agent_tool_parser import ToolParser, ToolError

parser = ToolParser(
    allowed_tools={"read_file", "write_file", "execute_command"},
    tool_aliases={"run": "execute_command", "view": "read_file"},
    param_aliases={"cmd": "command", "file": "path"},
    allow_shell_fallback=True,
)

call = parser.parse("```bash\npytest tests/\n```")
# Returns ToolCall(name='execute_command', args={'command': 'pytest tests/'})
```

---

## Performance Characteristics

Measurements conducted on Python 3.12 (AMD EPYC, single execution thread):

| Evaluation Scenario | Average Latency | Throughput |
| :--- | :---: | :---: |
| DSML Markup Extraction | **0.038 ms** | ~26,000 calls / sec |
| DeepSeek Delimiter Parsing | **0.024 ms** | ~41,000 calls / sec |
| Markdown & Malformed JSON Repair | **0.015 ms** | ~65,000 calls / sec |

---

## Automated Testing & CI

All pull requests and commits to `main` are continuously validated across runtimes and target architectures in GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

- **Python Multi-Version Matrix**: Python 3.9, 3.10, 3.11, 3.12, and 3.13 running `ruff`, `mypy`, and the complete 104-test suite under pure-Python fallback (`AGENT_TOOL_PARSER_NO_EXT=1`).
- **Rust Core & Multi-Language Bindings**:
  - Rust stable toolchain: `cargo fmt`, `cargo clippy -D warnings`, and 58 unit, integration, and benchmark tests.
  - C-ABI shared library compilation (`libagent_tool_parser_c.so`) and C unit tests.
  - Native Go binding integration tests (`go test -v ./bindings/go/...`).
  - WebAssembly compilation target verification (`wasm32-unknown-unknown` release build).
  - PyO3 native module build and dual-engine differential parity verification (`tests/test_parity.py` with 240+ assertions).
- **Storage-Conscious CI Architecture**: The entire test matrix runs in ephemeral runner memory with zero artifact storage overhead, preventing billing impact on account storage quotas.

---

## Contributing

We welcome contributions to `agent-tool-parser`. Because the project maintains dual implementations (pure-Python reference and compiled Rust core), all pull requests modifying parsing heuristics, format extractors, or cleaner rules must provide synchronized changes across both engines. If full parity is technically infeasible, a comprehensive technical rationale must be provided.

See our [**Contributing Guide (CONTRIBUTING.md)**](CONTRIBUTING.md) for full instructions, test commands, and submission checklists.

---

## Security & Execution Disclaimer

`agent-tool-parser` is strictly a **passive lexical and structural parser**. It does not validate, authenticate, sandbox, or execute tool calls, shell commands, or arbitrary parameters emitted by language models.

- **No Execution Safety Guarantees**: Language models can hallucinate, produce malformed payloads, or be manipulated via adversarial prompt injections. Parsed tool calls and arguments must **never** be executed directly without strict downstream input validation, schema enforcement, authorization checks, and process isolation.
- **Sandboxing & Privilege Boundaries**: The integrating application, agent framework, and operators are solely responsible for enforcing execution security, sandboxing (e.g., containers, VMs, seccomp/gVisor), rate limits, and human-in-the-loop approvals.
- **Limitation of Liability**: In accordance with the MIT License, this software is provided **"AS IS"**, without warranty of any kind, express or implied. In no event shall the authors, maintainers, or copyright holders (including the InBoost Team) be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, data loss, unauthorized system access, financial losses, service downtime, or hardware damage) arising in any way out of the use, interpretation, execution, or parsing anomalies of this software.

See our full [**Legal & Execution Disclaimer (DISCLAIMER.md)**](DISCLAIMER.md).

---

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 InBoost Team.

---

*Maintained by the **InBoost Team**.*

