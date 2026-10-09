## Final production-config validation (2026-10-09, addendum)

User-facing config (256K default context, 1024 default max-num-seqs, hermes tool parser +
auto tool choice added), everything else identical to the baseline above:

- Boot ~130 s (warm caches), `max_model_len=262144` confirmed
- Agent workload (4x12): run1 496.7 tok/s, run2 487.8 tok/s, 0 errors — **parity with the
  131K/128 baseline** (mean +1.1%), i.e. no measurable cost from the wider context/cap defaults
- MTP acceptance during agent windows: length 2.92-3.51, draft acceptance 64-84%
  (hot windows reach 83.6%, matching the cache-hit-raises-acceptance pattern)

Full log: `agent_final_run.log`.
