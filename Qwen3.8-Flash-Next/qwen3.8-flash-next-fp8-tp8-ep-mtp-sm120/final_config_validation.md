## Final production-config validation (2026-10-09, addendum)

User-facing config (256K default context, 1024 default max-num-seqs, qwen3_coder tool parser +
auto tool choice added), everything else identical to the baseline above:

- Boot ~130 s (warm caches), `max_model_len=262144` confirmed
- Agent workload (4x12): run1 496.7 tok/s, run2 487.8 tok/s, 0 errors — **parity with the
  131K/128 baseline** (mean +1.1%), i.e. no measurable cost from the wider context/cap defaults
- MTP acceptance during agent windows: length 2.92-3.51, draft acceptance 64-84%
  (hot windows reach 83.6%, matching the cache-hit-raises-acceptance pattern)

Full log: `agent_final_run.log`.


**Correction (2026-10-09)**: the validated config used `--tool-call-parser qwen3_coder`, not `hermes` — this model's template emits XML-style tool calls. See `tool_parser_ab_note.md` for the A/B evidence (hermes: 5/6 turns raw-XML leak, zero structured calls; qwen3_coder: clean). Throughput numbers above are unaffected (the parser only engages on requests carrying `tools`).