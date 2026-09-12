---
name: Bug Report
about: Create a report to help us improve agent-tool-parser
title: "[BUG] "
labels: bug
assignees: ''
---

### Describe the Bug
A clear and concise description of what the bug is.

### Engine Mode
- [ ] Pure Python (`AGENT_TOOL_PARSER_NO_EXT=1`)
- [ ] Accelerated PyO3 (`_accelerated`)
- [ ] Rust Core (`agent-tool-parser-core`)
- [ ] Go bindings / C-ABI / WASM

### Minimal Reproducible Example

```python
from agent_tool_parser import parse_tool_calls

raw_model_output = """
<paste sample model completion here>
"""

calls = parse_tool_calls(raw_model_output)
print(calls)
```

### Expected Behavior
A clear description of what tool calls, names, or arguments you expected to be parsed.

### Actual Output or Error
Include tracebacks or unexpected parsed outputs here.

### Environment
- OS: [e.g., Linux Ubuntu 24.04, macOS Sonoma, Windows 11]
- Python version: [e.g., 3.11.8]
- Rust version (if building from source): [e.g., 1.80.0]
- Library version: [e.g., 0.1.0]
