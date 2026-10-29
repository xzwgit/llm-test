## Tool-call parser A/B — hermes vs qwen3_coder (2026-10-09)

Same model/server config, only the parser differs. Workload: real `tools`-API agent loop
(3 tools, canned results, 6 turns; script `agent_tools_sim.py`).

| arm | tool_call_turns | raw XML leak | structured tool_calls | verdict |
|---|---|---|---|---|
| hermes run1 | 0/6 | **5/6 turns leak `<tool_call>`/`<function=` in content** | none | FAIL |
| qwen3_coder run1 | 3/6 | 0 | ok (`structured_tool_calls_ok=true`) | PASS |
| qwen3_coder run2 | 3/6 | 0 | ok | PASS |

- Why our earlier plain-text agent sim missed this: the parser only engages when the request
  carries `tools`. **Lesson (now a test standard): agent-scenario tests must include real
  tools-API rounds and assert structured `tool_calls` + no raw-XML leak.**
- Root cause of the wrong choice: model-family habit (Qwen -> hermes) instead of reading the
  model's own `chat_template.jinja` (XML format) and the parser registry (`qwen3_coder` ==
  `qwen3_xml` -> `Qwen3EngineToolParser`).
- MTP during tools runs (qwen3_coder arm): mean acceptance length 3.04-3.15, draft acceptance
  68-72%.
- Cross-checked with the official vLLM recipe for this model (`qwen3_coder` + `--reasoning-parser
  qwen3`; plain TP8 documented as incompatible with the FP8 checkpoint — matches our TP4/TP8
  boundary; MTP benchmarking officially recommended with SPEED-Bench rather than random).

Raw outputs incl. the first (harness-crashed) round and the fixed rerun: `tools_parser_ab.log`.
Standard script (also deployed at /root/vllm31/agent_tools_sim.py on .154/.156): `agent_tools_sim.py`.
