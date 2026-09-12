## Summary of Changes
Provide a brief explanation of what this pull request does and why it is needed.

## Dual-Engine Parity Verification
`agent-tool-parser` enforces strict Dual-Engine Parity between the pure-Python reference engine and the compiled Rust core.

- [ ] I have implemented all parsing/cleaning changes in both `src/agent_tool_parser/` and `crates/agent-tool-parser-core/`.
- [ ] *OR* If parity is not technically possible, I have included the mandatory **Dual-Engine Parity Analysis** section below.

### Dual-Engine Parity Analysis (only required if engines diverge)
*Root Cause, Impact Assessment, and Mitigation Strategy.*

---

## Contributor Legal & DCO Checklist
- [ ] All commits in this pull request are signed off (`git commit -s`) per the [Developer Certificate of Origin (DCO 1.1)](DCO.md).
- [ ] I confirm that this submission contains no proprietary, leaked, or license-incompatible code.
- [ ] I have read and agree to the [Contributing Guide](CONTRIBUTING.md) and Contributor Indemnification terms (Section 3.2.3).

---

## Test & Verification Checklist
- [ ] Unit tests added to both `tests/test_parser.py` (Python) and `crates/agent-tool-parser-core/` (Rust).
- [ ] `cargo test --all-targets` passes with zero errors.
- [ ] Pure-Python test suite passes: `AGENT_TOOL_PARSER_NO_EXT=1 python -m unittest discover -s tests`
- [ ] Accelerated PyO3 test suite passes: `python -m unittest discover -s tests`
- [ ] Rust formatting & linting: `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings`
- [ ] Python formatting & linting: `ruff check .` and `ruff format --check .` and `mypy src`
