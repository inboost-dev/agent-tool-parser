# Legal & Execution Disclaimer

**Last Updated:** September 2026

`agent-tool-parser` is strictly a **passive text extraction, syntax repair, and normalization library**. It transforms raw string outputs produced by large language models (LLMs) into structured data models.

---

## 1. No Execution & No Safety Sandboxing

- **Passive Extraction Only**: This library does **not** evaluate, sandbox, invoke, or execute shell commands, Python scripts, database queries, web requests, or any other tools.
- **Untrusted Input Warning**: Large language models are non-deterministic and can produce unintended, malformed, or hallucinated outputs, and may be manipulated via adversarial prompt injections.
- **Downstream Security Responsibility**: Downstream applications, agent frameworks, operators, and developers bear **sole and exclusive responsibility** for:
  1. Validating and sanitizing all tool names and parameter schemas before execution.
  2. Enforcing strict execution sandboxing (e.g., containers, virtualization, seccomp, gVisor, or restricted privilege boundaries).
  3. Implementing authorization policies, rate limits, and human-in-the-loop approvals for sensitive operations.

---

## 2. Limitation of Liability ("AS IS")

In accordance with the [MIT License](LICENSE):

1. **NO WARRANTY**: THE SOFTWARE IS PROVIDED **"AS IS"**, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, AND NONINFRINGEMENT.
2. **NO LIABILITY FOR AGENT ACTIONS**: IN NO EVENT SHALL THE AUTHORS, MAINTAINERS, CONTRIBUTORS, OR COPYRIGHT HOLDERS (INCLUDING THE INBOOST TEAM) BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH:
   - The use, interpretation, or parsing anomalies of this software;
   - Any action, command execution, or system modification performed by an autonomous agent or downstream application using data parsed by this software;
   - Any loss of data, unauthorized system access, financial loss, service interruption, or hardware compromise.

---

## 3. Contributor Intellectual Property & Indemnification

By contributing code, patches, or documentation to this project (under the [Developer Certificate of Origin (DCO 1.1)](DCO.md) and [Contributing Guidelines](CONTRIBUTING.md)):

- Each contributor represents and warrants that their contribution is original work (or under an OSI-approved permissive license) and does not infringe third-party patents, copyrights, trade secrets, or proprietary rights.
- Any contributor (and any organization on whose behalf a contribution is made) assumes sole and exclusive legal and financial liability for their submission and agrees to defend, indemnify, and hold harmless the project maintainers, organization, InBoost Team, and all downstream users against any third-party claims or damages resulting from their contribution.
