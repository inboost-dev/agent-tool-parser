# Security Policy

## 1. Supported Versions

We provide security updates and patches for the following versions of `agent-tool-parser`:

| Version | Supported          |
| :------ | :----------------- |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

---

## 2. Threat Model & Scope

`agent-tool-parser` is a **passive text extraction and normalization library**. It reads strings produced by language models and outputs structured data models (`ToolCall`).

### In Scope
The following issues are considered valid security vulnerabilities in `agent-tool-parser`:
- **Denial of Service (ReDoS)**: Regular expression catastrophic backtracking causing infinite loops or excessive CPU consumption on adversarial inputs.
- **Memory Safety Violations**: Memory corruption, buffer overflows, use-after-free, or undefined behavior in the Rust core, C-ABI (`agent-tool-parser-c`), PyO3 bindings, or WebAssembly artifacts.
- **Uncontrolled Resource Consumption**: Out-of-memory (OOM) crashes triggered by small inputs.
- **Parser Injection Flaws**: Scenarios where malformed tag nesting allows attacker-controlled payload injection that bypasses parser parameter extraction boundaries.

### Out of Scope
The following issues are **not** vulnerabilities in `agent-tool-parser`:
- **Unsafe Tool Execution**: `agent-tool-parser` does not execute shell commands, Python scripts, or API requests. Downstream applications and agent runtimes are solely responsible for sandboxing, schema validation, authorization, and safe execution of tools.
- **Model Prompt Injections**: If an LLM is manipulated via prompt injection into emitting a destructive tool call (e.g., `bash` with `rm -rf /`), that is an LLM application safety concern, not a parser vulnerability.
- **Denial of Service via Massive Inputs**: Processing inputs exceeding available system memory without appropriate agent framework stream limits.

---

## 3. Reporting a Vulnerability

If you discover a security vulnerability in `agent-tool-parser`, please do **not** open a public issue.

Instead, please disclose it responsibly via one of the following channels:

1. **GitHub Private Security Advisory (Preferred)**:
   Submit a report via the [Security Advisories page](https://github.com/inboost-dev/agent-tool-parser/security/advisories/new).
2. **Email**:
   Send a detailed vulnerability report to:
   - **boostcode.sg@gmail.com**

### Information to Include
To help us triage and resolve the issue quickly, please include:
- A detailed description of the vulnerability and its potential impact.
- Steps to reproduce, including minimal sample input text and configuration.
- Runtime environment (Python version, Rust version, OS architecture, accelerated vs. pure-Python mode).
- Proof-of-concept (PoC) code, if available.

---

## 4. Response Timeline & Disclosure Process

- **Initial Response**: We will acknowledge receipt of your vulnerability report within **48 hours**.
- **Assessment**: We will verify the issue and provide an estimated timeline for remediation within **5 business days**.
- **Fix & Release**: Security patches will be prioritized and published in a patch release.
- **Public Disclosure**: We coordinate public disclosure with the reporter once a fix and security advisory are available.
