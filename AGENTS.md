# Agent Operating Guidelines & Rules (AGENTS.md)
**Scope:** Repository-wide (`agent-tool-parser`)  
**Target:** AI Assistants, Antigravity, OpenHands, Claude, Cursor

---

## 1. Repository Mission & Architecture Invariants
`agent-tool-parser` provides high-performance, deterministic tool-call and AST extraction from LLM outputs.

1. **Dual-Engine Parity:** Any parsing, extraction, or regex change MUST be implemented synchronously in both the compiled Rust engine and the pure-Python fallback with identical semantics.
2. **Minimal Dependencies:** Zero transitive runtime bloat; keep dependencies lightweight and permissive.
3. **DCO & Verification:** All commits must be signed off (`git commit -s`). Test suites must pass across pure-Python mode (`AGENT_TOOL_PARSER_NO_EXT=1`) and compiled Rust bindings.

---

## 2. Mandatory Local Acceleration & Zero-Noise Tooling
To eliminate context token bloat, prevent message rate limit exhaustion, and maintain sub-millisecond developer feedback loops, all AI assistants and coding agents MUST leverage the following local tools:
- **Structural Search & Refactoring (`ast-grep` / `sg`):** Invoke `ast-grep run --pattern '<pattern>' [path]` instead of blind grep to match syntax nodes without noise.
- **Sub-Millisecond Rust Linting (`ruff` & `biome`):** Run `cargo clippy` and `ruff check [path]` (< 2 ms) to detect syntax errors and unused imports before submitting patches.
- **Repository Architectural Mapping (`Graphify` & `RepoMap`):** Run `repomap [path]` to inspect class/function signatures without loading full file bodies; query `graphify-out/graph.json` or run `graphify query "<question>"` for multi-module architectures.
- **Runtime Test Offload (`InBoost Local`):** Run tests via `inboost-local run <command>` to crush failures (-95%) and localize AST repair ranges.
