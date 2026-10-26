# DSV4.1-Flash-1017 on RTX 6000D 84G (8x SM120) — DSpark k=3/4/5 full A/B

**Date**: 2026-10-20 · **Box**: single node, 8x RTX 6000D 84G (SM120, 85651 MiB/card), driver 615.71.09, 2x NUMA

## Stack

- vLLM `0.32.0` release + torch 2.13.0+cu132 + FlashInfer 0.7.1rc5
- SM120 uniform-64 block-size patch: vllm-project/vllm#60762
- FULL cudagraph prefill-dispatch fix: vllm-project/vllm#62535
- Checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (48 shards, 476G), fp8 KV + fp8 indexer
- P2P: modprobe RegistryDwords configured (busbw 45 GB/s without NCCL_P2P_LEVEL=SYS)

## What's in here

| Path | Content |
|---|---|
| `docs/benchmark_report.html` | Full report: 18 zero-failure random tiers (k=3/4/5 × c1/c4/c16/c32/64K c2/c4) + 9 Agent sessions (3 k × 3 sessions × 12 turns) + k-value comparison summary |
| `logs/k_ab_158.log` | Raw `vllm bench serve` output, 18 tiers (official fields incl. per-position acceptance) |
| `logs/c_ramp.log` | k=3 concurrency ramp c1-c12 (1K/1K) |
| `logs/c_ramp_boot.log` | Server boot log for the ramp run |
| `logs/agent_k{3,4,5}_results.jsonl` | Agent session results (12-turn tools loop × 3 sessions per k) |
| `logs/agent_k{3,4,5}_spec.log` | Server-side SpecDecoding metrics windows for each Agent k leg |

## Key results

- All 18 random tiers + 9 Agent sessions: zero failures
- k=3: c1 101.6 / c32 966 / 64K-c4 444 tok/s
- k=4: c1 88.0 / c32 827 / 64K-c4 473 tok/s; **Agent optimal (1.32s/turn, 3/3 clean)**
- k=5: c1 137.3 / c32 785 / 64K-c4 542 tok/s; Agent slowest (1.71s/turn)
- DSpark acceptance length at 64K context: 3.2-4.7 (vs 2.0-2.5 at 1K), confirming higher speculation benefit for long contexts
- k=5 shows bimodal acceptance in Agent (short tool-response turns near-full, long generation drops to ~3.1)

## Notes

- 84G cards have ~18G/card remaining after weights (~59.5G), yielding ~11-12M token KV pool: single-stream 1M works, concurrent 1M limited to ~2-3
- FlashInfer JIT on this box requires `PATH` to include `/usr/local/cuda/bin` (nvcc) and system `ninja`; the FlashInfer SM120 specialization check fails without both
- k=3 Agent had 1 raw XML leak (`<tool_call>` text instead of structured `tool_calls`) in 1/36 turns; k=4/5 had none


## Unified matrix retest (2026-10-21, supersedes the k345 A/B above)

Same env (vLLM 0.32.0 + FlashInfer 0.7.2 + #62535) with a **uniform matrix**: per-k independent boot, 5 length groups (in=out 1K/4K/8K/32K/64K) x concurrency ladders + agent-tools tier (3 sessions x 12 turns per k). k=3/4/5, 87 random tiers + 9 agent sessions, zero failures.

- util 0.90 rule + capture=128x(k+1) after the k=4 OOM incident (0.92 + capture 768 left only ~200MB headroom at 1K c128; evidence: `logs/unified-matrix/matrix_k4_failed_v1.log`, engine CUDA-OOM at the c128 peak, 21 subsequent tiers fast-fail)
- 1K c1 double-run per k (cold-start first run silently dropped, same label last-wins)
- 8K/8K c32 outlier verdict: k=4 low value is REAL (879/883/919 across three samples), k=3 run-to-run variance ±12% (`logs/unified-matrix/c32_redo.log`)
- Agent-window speculative acceptance from server-side SpecDecoding windowed counters: k3 60.3% (acclen 2.83) / k4 50.8% (3.03) / k5 55.1% (3.76); consistency check acclen ~= 1 + rate x k holds
- Full report: `docs/DSV4.1-Flash-1017-PRO6000D-FlashInfer072-DSpark-k345-unified-matrix.html`
- Raw logs carry the full BOOT CMD line (api-key redacted)


## Multimodal enabled (2026-10-22)

Same recipe minus `--language-model-only` (vision tower loaded; image modality active, video disabled by model spec). Boot ~90-126s, zero OOM at util 0.90/131K/DSpark k=3; KV pool essentially unchanged vs text-only (6,255,239 tokens - vision costs ~0.04% of the pool).

- E2E graded test (`vl_e2e.py`): image 512/1024/2048 squared x c1/c4, red-circle + text-in-image graded -> **15/15 pass**. TTFT 0.12s (512) -> 0.19s (2048); c4 concurrency does not raise TTFT materially.
- Official `vllm bench mm-processor` (TP8; note TP=1 default cannot fit the 476G model - OOM evidence in `mmproc.log`): HF processor mean 8.9ms / encoder forward mean 28.5ms (P99 43.8ms) across 30 random-mm prompts at three resolutions. Requires pandas in the venv for the results table (`vllm[bench]`).
- Raw: `logs/multimodal/`

## 2026-10-26 · no-spec full unified matrix (`logs/nospec-matrix/`)

Baseline leg without speculative decoding (no DSpark draft), same boot config as k345-v3 minus `--speculative-config`.

| Path | Content |
|---|---|
| `docs/DSV4.1-Flash-1017-PRO6000D-FlashInfer072-nospec-unified-matrix.html` | Full 29-tier report (no-spec, acceptance cols = —) |
| `logs/nospec-matrix/tiers.log` | Raw bench output, 29 tiers + c1 double-run, BOOT CMD in header |
| `logs/nospec-matrix/boot.log` | Server boot log |
| `logs/nospec-matrix/agent_s{1,2,3}.out` | Agent 12-turn tools sessions (s2: 1 raw-XML leak, errors=0) |
| `logs/nospec-matrix/vl_e2e.out` | Multimodal E2E, all cases passed |
| `logs/nospec-matrix/matrix_nospec.sh` | Driver script |

Key: vLLM 0.32.1rc1.dev463 + FlashInfer 0.7.2.post1, boot ~90s, 2h29m total, zero failures / zero anomalies across 30 benches. Single-stream ~73-80 tok/s flat across 1K-64K; aggregate 1K/1K c128 = 2562 tok/s.