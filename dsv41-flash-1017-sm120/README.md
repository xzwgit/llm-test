# DeepSeek-V4.1-Flash-1017 on SM120 (8x RTX PRO 6000) — full benchmark + phase-probe suite

**Date**: 2026-10-19 · **Box**: single node, 8x RTX PRO 6000 Blackwell Server Edition 96G (SM120), 2x EPYC 9A75, driver 595.71.05

## Stack

- vLLM `0.32.1rc1.dev24+ge719b41e2` (latest main nightly) + torch 2.14.0+cu132 + FlashInfer 0.7.1rc5 (cubin left at the shipped 0.7.0.post1)
- SM120 uniform-64 block-size patch: vllm-project/vllm#60762 (PR head cf63b064, byte-identical overlay)
- Checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (48 shards, 476G), DSpark spec=3, fp8 KV + fp8 indexer, engram cpu_offload, TP8+EP, 1M context, FULL cudagraphs (capture 512)
- `NCCL_P2P_LEVEL=SYS` required on this platform (busbw 22 -> 44 GB/s without/with)

## What's in here

| Path | Content |
|---|---|
| `docs/benchmark_report.html` | Full report: 25 zero-failure tiers (1K ladder c1-c128, long-context 32K-1M, 128K concurrency, 64K/64K ramp c2-c12), per-tier DSpark acceptance, phase-probe suite (flip + needle), NVFP4-KV probe, correctness, metric glossary |
| `logs/v41_1017_full.log` | Raw `vllm bench serve` output, all 25 tiers (official fields incl. per-position acceptance) |
| `logs/v41_1017_boot.log` | Server-side log for the suite run |
| `logs/needle_results.jsonl` | 256-prompt needle phase probe results (per-residue accuracy) |
| `logs/agent_k_ab.log` | DSpark k=3/4/5 agent A/B attempts — captured mid-investigation; see the spec-decode finding below |

## Key results

- 1K ladder: c1 125.2 -> c128 **2,258.8 tok/s**, zero failures across all 25 tiers
- Long context: 32K-1M single-stream all clean; 1M TTFT 36.3 s, TPOT 9.5 ms
- 64K/64K ramp: c2 359.6 -> c12 **1,141.7 tok/s** (acceptance length 3.4-3.7, well above the 1K tiers' 1.7-2.3)
- NVFP4 KV (`nvfp4_ds_mla`): architecturally rejected on SM120 — `ValueError: nvfp4_ds_mla needs the V4.1 KV records, which FlashInfer decodes only on SM100.`

## Phase-probe suite (flip + needle, on 1017)

flip: no 8/32 oscillation at the calibrated cursor (P(8)/P(32) outside top-20). needle: **255/256 = 99.6%**, flat across all 8 residues, even/odd gap ≈ 0 — the phase signature reported for the original V4.1-Flash weights does not reproduce on 1017.

## Spec-decode finding (filed as vllm-project/vllm#62417)

DSpark speculative decoding corrupts thinking/chat outputs on the 1017 checkpoint (degenerate reasoning loops, no structured tool calls) at k=3/4/5/7 on both v0.32.0 and the nightly — while the same boot **without** speculation is clean (12/12 agent turns, structured tool calls, 323) and the original V4.1-Flash checkpoint with DSpark k=3 is clean too. Plain-text `/v1/completions` is unaffected (all 25 tiers above ran with speculation on). Workaround: disable speculation for chat/agent serving on 1017.
