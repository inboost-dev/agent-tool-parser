# Contributing to agent-tool-parser

Thank you for your interest in contributing to `agent-tool-parser`. We welcome bug reports, format extensions, performance optimizations, and documentation improvements.

To maintain architectural stability, cross-platform portability, and legal integrity across all supported runtimes (Python, Rust, Go, C/C++, WebAssembly), all contributions must comply with the guidelines below.

---

## 1. The Dual-Engine Synchronization Policy

`agent-tool-parser` maintains two synchronized core implementations:
1. **Pure-Python Reference Engine** (`src/agent_tool_parser/`): Universal fallback that executes in constrained Python environments without requiring a C/Rust compiler.
2. **Compiled Rust Core Engine** (`crates/agent-tool-parser-core/`): High-throughput native engine powering PyO3 Python bindings, Go (via C-ABI / `cgo`), C/C++, and WebAssembly.

### The Parity Rule
**Any pull request modifying extraction logic, format delimiters, cleaner heuristics, or normalization rules MUST implement the change in both engines simultaneously.**

Specifically:
- If you add or modify a parsing pattern in `src/agent_tool_parser/parser.py`, you must implement the identical logic in `crates/agent-tool-parser-core/src/parser.rs`.
- If you adjust JSON cleaning, parameter repair, or entity unescaping in `src/agent_tool_parser/cleaners.py`, you must make the corresponding adjustment in `crates/agent-tool-parser-core/src/cleaners.rs`.
- Corresponding unit tests must be added to both `tests/test_parser.py` (Python) and `crates/agent-tool-parser-core/src/lib.rs` (Rust).

### Exception Criteria (When Parity Is Not Possible)
If implementing identical behavior across both engines is technically infeasible (for example, due to language-specific standard library capabilities such as Python's `ast` module vs. Rust tokenizers), the pull request **must include a dedicated section in the PR description titled "Dual-Engine Parity Analysis"**:

The analysis must clearly and technically document:
1. **Root Cause**: The specific runtime, architectural, or language constraint preventing parity.
2. **Impact Assessment**: Which formats or inputs behave differently between engines, and why this does not break downstream contract guarantees.
3. **Mitigation Strategy**: The fallback or approximation implemented in the counterpart engine to ensure graceful handling.

Pull requests introducing discrepancies without this explicit rationale will not be merged.

---

## 2. Pragmatic Dependency Policy

We reject dogmatic "zero-dependency purism" when it forces the project to reinvent complex, fragile wheels (such as ad-hoc JSON repair state machines or character encoding decoders). However, we strictly prevent **transitive dependency bloat and supply-chain risk**.

### Criteria for Introducing Dependencies
Before proposing an external library in either Python or Rust, verify that it meets the following requirements:

1. **Zero or Minimal Transitive Footprint**:
   The dependency must be focused and self-contained (0–1 sub-dependencies). Libraries pulling large dependency trees (e.g., LangChain, BeautifulSoup, Transformers) will be rejected.
2. **Zero Cold-Start Penalty**:
   In Python, the library's import overhead must not exceed 5–10 milliseconds. In Rust, it must not substantially degrade build times or binary size.
3. **Permissive Licensing Only**:
   Must use an OSI-approved permissive license (MIT, Apache-2.0, BSD, ISC) fully compatible with downstream MIT redistribution.
4. **Cross-Platform & Portability**:
   Dependencies must compile cleanly on Linux (glibc and musl), macOS, and Windows. In Rust, core dependencies must support WebAssembly (`wasm32-unknown-unknown`).
5. **Graceful Degradation**:
   In Python, non-standard runtime optimizations (such as `orjson` or `json-repair`) must be declared as optional extras in `pyproject.toml` with clean standard library fallbacks in pure-Python mode.

---

## 3. Legal Purity, IP Hygiene & Licensing

To protect users, enterprises, and downstream integrators, `agent-tool-parser` enforces strict intellectual property (IP) hygiene.

### 3.1 Plain-English Summary (TL;DR for Developers)

Before reviewing the binding legal text below, here is the short, plain-English summary:

| Dimension | Rule | Practical Meaning |
| :--- | :--- | :--- |
| **Original & Open Source Code** | Permissive only | Write original code or use code with permissive licenses (MIT, Apache 2.0, BSD). Always preserve original copyright notices. |
| **Model Outputs & Interoperability** | Allowed & Encouraged | Inspecting, analyzing, and parsing publicly observable model outputs (DeepSeek, Claude, Qwen, etc.) for parser interoperability is legitimate and standard practice. |
| **Prohibited Materials** | No Incompatible or Leaked Code | Never submit code with licenses incompatible with MIT, proprietary code, materials from internal/closed repos, or anything under NDA. |
| **Commit Sign-Off** | Mandatory (`git commit -s`) | Every commit must include `Signed-off-by:` via the `-s` flag, confirming you have the legal authority to contribute. |
| **Liability & Indemnification** | Contributor & Employer on the hook | If you submit infringing, stolen, or tainted code, **you and your organization/employer** (jointly and severally) bear sole legal and financial responsibility to defend and indemnify the project and its users against all claims. |
| **Outbound License** | MIT License | All contributions are permanently licensed under the project's standard MIT License. |

---

### 3.2 Formal Legal Terms (Binding)

#### 3.2.1 Developer Certificate of Origin (DCO 1.1)
All commits must be signed off by the author in accordance with the Linux Foundation Developer Certificate of Origin (DCO) version 1.1.

Add a sign-off line to each commit message using git's `-s` / `--signoff` flag:
```bash
git commit -s -m "fix(dsml): handle unquoted attribute values"
```

The resulting commit message will contain:
```text
Signed-off-by: Random J Developer <random@developer.example.org>
```

By adding this line, you certify the statement defined in [**Developer Certificate of Origin version 1.1 (DCO.md)**](DCO.md).

#### 3.2.2 Intellectual Property Hygiene & Interoperability
- **Clean-Room Interoperability**: `agent-tool-parser` extracts and normalizes publicly observable data serialization formats (e.g., XML tags, DSML markup, JSON schemas, public HTTP streaming tokens) solely for technical interoperability. Observing, analyzing, and parsing public model completions, wire formats, and documented protocols is legitimate and standard practice.
- **Prohibited Materials**: Under no circumstances may a contributor submit:
  - Decompiled, disassembled, or leaked proprietary source code from closed SDKs or internal repositories.
  - Materials subject to non-disclosure agreements (NDAs), trade secrets, or third-party confidentiality obligations.
  - Code subject to licenses or terms incompatible with downstream MIT licensing and unrestricted commercial redistribution.
- **AI-Assisted Contributions**: Contributors using generative AI tools bear full responsibility for verifying that generated code is free from proprietary, confidential, or license-incompatible third-party source material.

#### 3.2.3 Contributor Representations, Warranties & Indemnification
For the purposes of this policy, **"Contributor"** encompasses any individual, corporation, partnership, association, or other legal entity submitting code, patches, documentation, or other materials, as well as any employer, principal, or organization on whose behalf the contribution is submitted.

By submitting any pull request, patch, or commit (and signing off via DCO 1.1), the Contributor explicitly represents, warrants, and agrees to the following terms:

1. **Right and Authority to Contribute**: You possess the full legal right, power, and corporate or organizational authority to submit the contribution under the project's [MIT License](LICENSE), unencumbered by any third-party rights, employment restrictions, work-for-hire agreements, or contractual obligations.
2. **Non-Infringement**: Your contribution is original work (or properly attributed permissively licensed open source) and does not infringe, misappropriate, or violate any patent, copyright, trademark, trade secret, or other proprietary or contractual right of any person or entity.
3. **Sole Legal & Financial Liability**: In the event that your contribution infringes any intellectual property, breaches any confidentiality agreement, or violates any applicable law, regulation, or third-party terms of service, **the Contributor (together with any entity, employer, or principal on whose behalf the contribution is made, jointly and severally) assumes sole and exclusive legal and financial liability**.
4. **Full Indemnification & Hold Harmless**: The Contributor and any associated entity agree to defend, indemnify, and hold harmless the project maintainers, organization, InBoost Team, and all downstream users and integrators from and against any and all claims, demands, lawsuits, regulatory investigations, liabilities, damages, losses, judgments, settlements, penalties, and expenses (including reasonable attorneys' fees, litigation costs, and enforcement expenses) arising out of or resulting from the contribution.

#### 3.2.4 License Grant
By submitting a pull request, you agree that your contribution is irrevocably licensed under the repository's [MIT License](LICENSE).

---

## 4. Testing & Verification Requirements

Before submitting your pull request, verify that all test suites pass cleanly across all supported runtimes:

### 1. Rust Workspace Test Suite
Verify that all crates (core, C-ABI, PyO3, WASM) compile without warnings and pass unit tests:
```bash
cargo test --all-targets
```

### 2. Python Pure-Python Fallback Suite
Verify that the reference pure-Python implementation passes all tests without native acceleration:
```bash
AGENT_TOOL_PARSER_NO_EXT=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

### 3. Python Native Acceleration Suite
Compile the release extension and verify that the accelerated engine passes all tests:
```bash
cargo build --release -p agent-tool-parser-py
cp target/release/lib_accelerated.so src/agent_tool_parser/_accelerated.so
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

### 4. Go Binding Suite (if C-ABI or headers were modified)
If changes affect `agent-tool-parser-c` or `include/agent_tool_parser.h`:
```bash
cargo build --release -p agent-tool-parser-c
cd bindings/go/agenttoolparser
LD_LIBRARY_PATH="../../../target/release:$LD_LIBRARY_PATH" go test -v
```

---

## 5. Code Style & Linting

### Python
- Formatted with **Ruff** (`line-length = 100`).
- Strict typing checked with **mypy**.
```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
```

### Rust
- Formatted with **rustfmt**.
- Linted with **clippy** with no warnings:
```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

---

## 6. Pull Request Checklist

Please ensure your pull request adheres to the following checklist:

- [ ] Parsing logic changes are implemented in both `src/agent_tool_parser/` (Python) and `crates/agent-tool-parser-core/` (Rust).
- [ ] If parity is not possible, a detailed **Dual-Engine Parity Analysis** is included in the PR description.
- [ ] Any new dependencies adhere to the **Pragmatic Dependency Policy** (permissive license, zero transitive bloat, minimal cold-start impact).
- [ ] All commits are signed off (`git commit -s`) per the **Developer Certificate of Origin (DCO 1.1)**.
- [ ] You represent and warrant that the submission is clean of third-party IP/confidentiality violations and agree to the **Contributor Indemnification** terms (Section 3.2.3).
- [ ] The submission contains no proprietary, leaked, or license-incompatible code.
- [ ] Unit tests are added to both `tests/test_parser.py` and `crates/agent-tool-parser-core/src/lib.rs`.
- [ ] `cargo test --all-targets` passes with zero errors and zero warnings.
- [ ] `python3 -m unittest discover -s tests` passes under both `AGENT_TOOL_PARSER_NO_EXT=1` and native acceleration.
- [ ] Linters (`ruff`, `cargo clippy`) pass with zero warnings.
- [ ] Commit message follows conventional commit format (e.g., `feat: ...`, `fix: ...`, `docs: ...`).
