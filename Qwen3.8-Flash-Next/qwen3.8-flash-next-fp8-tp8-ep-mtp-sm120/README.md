# Qwen3.8-Flash-Next-FP8 — TP8+EP+MTP baseline, 8x RTX PRO 6000 (SM120)

2026-10-09, single node, vLLM official mainline.

## Environment
- vLLM `0.31.1rc1.dev158+gab905a885` + FlashInfer `0.7.1rc5` + DeepGEMM `2.8.0+1e1842a`
- FlashInfer autotune **enabled**, with the independent-mode patch from flashinfer PR #6161 applied
  to the rc5 install (per-rank profiling, no tuning collectives; boots non-blocking)
- 8x RTX PRO 6000 (SM120, 96GB), driver 595.71.05 + CUDA 13.2

## Boot command (exact)
```
vllm serve /data/models/Qwen3.8-Flash-Next-FP8   --served-model-name qwen3.8-flash-next --trust-remote-code   --host 0.0.0.0 --port 8123 --api-key ***   --tensor-parallel-size 8 --enable-expert-parallel   --speculative-config '{"method":"mtp","num_speculative_tokens":3}'   --max-model-len 131072 --max-num-seqs 128
# env: VLLM_USE_RUST_FRONTEND=1 FLASHINFER_AUTOTUNE_INDEPENDENT=1 HF_HUB_OFFLINE=1
```
Cold start ~2 min (warm compile caches); correctness 17x19=323; MTP draft
acceptance 75% at first request (mean acceptance length 3.25, per-position 0.938/0.750/0.562).

## Standard matrix (vllm bench serve, 1K in / 1K out, random + ignore-eos)

| tier | output tok/s | mean TTFT | median TPOT | bench accept len |
|---|---|---|---|---|
| c1 | 86.78 | 95.6 ms | 10.46 ms | 1.39 |
| c2 | 149.08 | 398.3 ms | 10.14 ms | 1.74 |
| c4 | 348.60 | 284.9 ms | 8.38 ms | 2.26 |
| c8 | 697.76 | 295.5 ms | 9.19 ms | 2.53 |
| c16 | 1186.39 | 375.1 ms | 10.49 ms | 2.49 |
| c32 | 1983.96 | 326.6 ms | 12.55 ms | 2.58 |

Note: random-token acceptance rises with concurrency (12.9% -> 52.7% draft acceptance,
c1 -> c32) — the inverse of the low-concurrency-favors-spec pattern we measured on
DeepSeek-V4.1-Flash+DSpark on the same hardware class.

## Agent workload (4 sessions x 12 turns, growing tool-result context, max_tokens 200)
- run1: wall 14.69s, 6990 out tokens, aggregate 475.8 tok/s, 0 errors
- run2: wall 13.75s, 6847 out tokens, aggregate 497.8 tok/s, 0 errors
- server-side spec stats during agent windows: mean acceptance length 3.27-3.38,
  draft acceptance 75-79% (structured short outputs accept far better than random,
  consistent with prior MTP/DSpark campaigns)

## Pure-TP boundary (same stack, evidence logs included)
- TP4 without EP: works — FP8 loader refines block scales `[128,128] -> [32,32]`
  to fit the sharded intermediate size 160 (`fp8.py:381`); 17x19=323, ~122 tok/s no-MTP
  (`puretp_tp4_test.log`)
- TP8 without EP: rejected at load — `gate's and up's weight = 80 is not divisible by
  weight quantization block_n = 128` (`noep_tp8_fail_error.txt`)
- => at TP8 granularity EP is mandatory for this checkpoint; upstream notes:
  vllm#59642 comments (issuecomment-6074237842 / -6074343657)

## Files
- `ab_ep_run.log` — full bench + agent outputs (EP arm) and NOEP boot failure record
- `puretp_tp4_test.log` — TP4-noEP boot log incl. scale-refinement evidence
- `noep_tp8_fail_error.txt` — TP8-noEP load rejection excerpt

Served state left running: TP8+EP+MTP @ 8123 (host 10.10.3.154).
