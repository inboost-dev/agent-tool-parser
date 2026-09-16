# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-09-16

### Added
- **MiniMax / LLM Prompt-Prefill Truncated JSON Prefix Recovery (ParseGuard Layer)**:
  - Added support for repairing truncated JSON prefixes commonly dropped by completion/prefill models (MiniMax-M3, DeepSeek-V3/V4, Qwen-2.5-Coder), including:
    1. Omitted opening curly brace before key: `tool": "bash", ...`, `name": "bash", ...`, `action": "read_file", ...`.
    2. Omitted brace and opening quote: `"tool": "read_file", ...`, `"name": "bash", ...`.
    3. Omitted brace and key name (direct colon with value): `": "read_file", ...`, `": "search", ...`, `: "read_file", ...`.
    4. Opening brace present with omitted key name: `{": "read_file", ...}` and `{ ": "search", ...}`.
    5. Truncated JSON calls preceded by Chain-of-Thought / reasoning text.
    6. Truncated JSON calls inside Markdown codeblocks (```` ```json ````).
- **Two-Tier Parser Architecture (`ToolParser`)**:
  - **Tier 1 (Fast-Path / Strict Spec Compliance)**: validates standard markdown codeblocks, RFC-8259 arrays, and balanced JSON objects first.
  - **Tier 2 (ParseGuard Fault-Tolerant Recovery)**: activated deterministically only when Tier 1 yields no valid calls, reconstructing headless JSON candidates with bracket-depth tracking and strict validation.
- **Dual-Engine Prefix Normalization**:
  - Added `normalize_truncated_json_prefix` in both Rust Core (`cleaners.rs`, zero-allocation `Cow<str>`) and Python Fallback (`cleaners.py`), applied transparently in `clean_json_str` and `extract_json_objects`.
  - Full binary parity verified in `test_parity.py` and benchmarked at <0.05ms (4.4 µs – 25 µs) with 0% false positives on standard text.

## [0.1.2] - 2026-09-13

### Fixed
- **DSML & XML Mismatched Parameter Closing Tag Robustness**:
  - Resolved classical ParseGuard edge case where models (such as MiniMax-M3, DeepSeek R1/V3) confuse `<tool_name>` with `<parameter>` and generate mismatched closing tags like `</｜DSML｜tool_name>` or `</tool_name>` instead of `</｜DSML｜parameter>`.
  - Added `PARAM_END_FALLBACK_RE` in both Rust core (`agent-tool-parser-core`) and Python fallback engines, safely stripping hallucinated closing tags from argument values without truncating outer invoke wrappers or compromising CDATA blocks.
  - Added full test coverage in Rust (`test_degenerate_dsml_parameter_typo_closing_tag`) and Python.

## [0.1.1] - 2026-09-12

### Changed
- Upgraded Python build backend to `maturin` with `abi3-py39` multi-platform wheels (Linux, macOS, Windows).
- Enabled automated cross-platform releases and Go proxy auto-registration in GitHub Actions.

## [0.1.0] - 2026-09-12

### Added
- **Dual-Engine Architecture**:
  - Pure-Python zero-dependency fallback engine with transparent native acceleration detection.
  - High-throughput Rust core engine (`agent-tool-parser-core`) with sub-millisecond execution.
  - PyO3 native module (`_accelerated`) with zero-copy Python deserialization.
  - C-ABI dynamic library (`agent-tool-parser-c`) and standard C header (`agent_tool_parser.h`).
  - WebAssembly bindings (`agent-tool-parser-wasm`) for browser and Node.js runtimes.
  - Native Go binding module (`bindings/go/agenttoolparser`) using C-ABI via cgo.
- **Heterogeneous Format Support**:
  - OpenAI function call objects and `tool_calls` arrays (including multi-call batches).
  - Anthropic Claude XML tags (`<tool_use>`, `<invoke>`, `<ant_tool_use>`) and parameter markup.
  - DeepSeek DSML tags (`<｜DSML｜invoke>`) with unquoted and quoted attribute syntax.
  - DeepSeek native streaming delimiter tokens (`<｜tool call begin｜>`).
  - Qwen-Agent tokens (`<|action_start|>`, `<|action_name|>`, `<|action_args|>`).
  - ChatGLM function call formats (`✿FUNCTION✿`).
  - Llama 3.x Python AST expressions (`call:tool(arg=val)` and positional argument mappings).
  - ReAct agent text patterns (`Action: ... Action Input: ...`).
- **Resilient Cleaners & Heuristic Repairs**:
  - Bracket-stack depth tracking for truncated JSON payloads, automatically balancing unclosed nested objects `{` and arrays `[` at stream EOF.
  - Single-quoted JSON repair preserving inner double quotes.
  - Two-step reasoning/thinking block extraction (`<think>...</think>`, `<thought>`, `<reasoning>`), safely stripping internal reasoning and preventing premature tool execution.
  - CDATA unwrapping (`<![CDATA[...]]>`) preserving whitespace, indentation, and unified diffs.
  - HTML entity unescaping (`&lt;`, `&gt;`, `&amp;`, `&quot;`, `&#39;`).
- **Verification & Parity**:
  - 98 Python unit tests passing on both pure-Python and accelerated engines.
  - 56 Rust core integration tests passing with 100% format parity.
  - 25 dedicated pathological, degenerate, and adversarial input tests.
  - Complete ReDoS resilience test with 50,000 repeating characters executed in <10ms.
- **Project Infrastructure**:
  - Developer Certificate of Origin (DCO 1.1) enforcement policy and contributor indemnification.
  - PEP 561 typing compliance (`py.typed`).
  - GitHub Actions CI/CD workflows for automated multi-runtime testing and multi-platform binary releases.
