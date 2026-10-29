# sm120opt-162 — xzwgit/vllm `sm120-opt` fork line on 10.10.3.162 (8×RTX 6000D 84G)

Campaign 2026-10-24: bootstrap the fork kernel line + first experiment
(**b12x MoE backend** for DeepSeek-V4.1-Flash-1017), per the DeepInfer
absorption plan in `SM120-OPT.md` on branch `sm120-opt`.

## Environment

- venv `/root/sm120opt` = vLLM `0.32.1rc1.dev451+g5f63ea461` (latest
  nightly via Tsinghua mirror) + FlashInfer 0.7.2 wheel + venv ninja;
  source checkout `/root/sm120opt_src` (branch `sm120-opt`).
- Weights: `/data/models/DeepSeek-V4.1-Flash-1017` (pre-existing,
  `/root/DeepSeek-V4.1-Flash-1017` is a symlink).
- Boot form: TP8 (no EP), kV fp8, DSpark k=7 probabilistic+block,
  PIECEWISE graphs, util 0.90, capture 1024, ctx 131072.
- Two patches carried by the fork line: **#60762 port** (SM120 block
  declaration [64] in `indexer.py` + `flashinfer_sparse.py`, ported to
  dev451 via `port_60762_dev451.py`; upstream PR still open) and the
  toolchain-PATH boot env (no system ninja on this host).

## Results timeline (all UTC 2026-10-24)

| run | config | outcome |
|---|---|---|
| b12x v2/v3 ep-leg | b12x + `--enable-expert-parallel` | hard reject: `B12X_MXFP4_BF16 does not support ... ep_size=8` (b12x is TP-only) |
| b12x v3 noep-leg | b12x TP8, stock dev451 | dies `attention.hpp:166 block_kv==32 or 64` → **#60762 still required on latest main** (dev451) |
| b12x v4 noep-leg | b12x TP8 + #60762 port | **READY 300s**, backend line `Using 'B12X_MXFP4_MXFP8' Mxfp4 MoE backend`; c32 1K×1K **1044.6 tok/s** (vs k7+DEEPGEMM 1000.4, +4.4%), c1 **321.5 tok/s** (+18.8%) |
| b12x v5 quality | same boot + probe suite | chat/zh/fact/tool-call/code OK, but completions-math degenerates (`32319=32319=...` loop) |

Earlier same-day: v2 legs failed on missing toolchain PATH (FlashInfer
silently disabled — "flashinfer-cubin is not installed / nvcc or ninja
missing"), fixed by venv ninja + PATH/CUDA_HOME; the error message then
misleads toward "needs flashinfer#4380" which was in fact present.

## Verdict

**b12x on SM120 × DSV4.1-1017: fast but numerically untrustworthy.**
Throughput wins (single-stream +18.8%) but exact-continuation tasks
silently corrupt while semantic chat passes — same failure family as the
`flashinfer_cutlass` MoE corruption (vllm#63473), lighter. Not viable as
a default route; fork work item: audit the b12x weight layout / scale
packing for DSV4.1 MXFP4 experts (layout undocumented per the DeepInfer
campaign). k=7 probabilistic+block sampling is **not** the cause: the
DEEPGEMM leg with the identical spec recipe passes the same probe
cleanly (see k7full campaign on .158).

## Note

`b12xv5_quality.json` holds the raw probe outputs; boot logs contain the
full JIT/autotune traces. API keys masked `***` in archived copies.
