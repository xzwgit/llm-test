# DSV4.1-Flash-1017 on RTX 6000D 84G (GB202 + EPYC 9654, SM120) x8 TP8+EP8

Third topology for the 1017 unified-matrix series (after the PRO 6000D 84G server:
`../dsv41-flash-1017-6000d-sm120/`). Same four-leg recipe: DSpark k3/k5/k7 + no-spec,
29 tiers per leg + c1 double-run + agent 3x12 turns + multimodal E2E.

Stack: vLLM 0.32.1rc1.dev463+gc9dbc9263 (nightly, patched) + FlashInfer 0.7.2.post1,
PIECEWISE CUDA graphs, kv fp8, util 0.90, 1M ctx, multimodal boot ([VL ON]).

## 2026-10-28 · four-leg matrix (`logs/k357-matrix/`)

| Path | Content |
|---|---|
| `docs/DSpark-k{3,5,7}-unified-matrix-gb202.html` | Spec-decode leg reports (19-col unified format) |
| `docs/no-spec-unified-matrix-gb202.html` | No-spec leg report |
| `logs/k357-matrix/matrix_{k3,k5,k7,nospec}.log` | Raw bench output, 30 blocks per leg (29 tiers + c1 double-run) |
| `logs/k357-matrix/spec_k{3,5,7}.log` | Engine-side SpecDecoding windows |
| `logs/k357-matrix/master.log` | Driver trace incl. BOOT CMD lines |
| `logs/k357-matrix/boot_*.log` | Server boot logs per leg (host addr sanitized) |
| `logs/k357-matrix/agent_*_s{1,2,3}.out` | Agent 12-turn tools sessions per leg |
| `logs/k357-matrix/vl_e2e*.out` | Multimodal E2E (spec phase on k3 engine + no-spec leg) |
| `logs/k357-matrix/matrix_192.sh`, `nospec_192.sh` | Driver scripts |

Key facts: KV pool per leg 11.45M / 9.89M / 8.99M / 13.08M tokens (k3/k5/k7/no-spec,
capture 512/768/1024 per the 128x(k+1) rule); zero failures across all four legs
(120 bench blocks); all four legs completed 11:56-19:08 UTC 2026-10-28.
