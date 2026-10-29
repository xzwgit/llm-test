# DeepSeek-V4.1-Flash on vLLM dev511 Nightly (SM120): PD Disaggregation, 1M Context, Full Concurrency Suite

> Full test results after upgrading from vLLM dev382 to **dev511** (includes mHC+AllReduce fusion #57643, DeepSelect #56464, and the FlashInfer JIT header fix).
> 中文版：[README.zh.md](README.zh.md)

## Key Finding: #57292 patch is STILL REQUIRED

dev511 upstream does **not** include the SM120 64-token page geometry. Without the patch from [#57292](https://github.com/vllm-project/vllm/pull/57292) (4 files: `indexer.py`, `flashinfer_sparse.py`, `sparse_mla.py`, `attention.py`), dev511 fails on SM120 with `block_kv == 32 or block_kv == 64` assertion. We confirmed this by upgrading a second machine (.153) with clean upstream code — it failed until the patch files were copied from the working machine (.154).

## FlashInfer JIT Fix (required for dev511)

The FlashInfer wheel does not bundle CUTLASS/CCCL headers needed for SM120 JIT compilation. Fix:

```bash
FI=<site-packages>/flashinfer/data
ln -sf <flashinfer-source>/3rdparty/cutlass $FI/cutlass
ln -sf <flashinfer-source>/3rdparty/cccl   $FI/cccl
ln -sf <flashinfer-source>/3rdparty/spdlog $FI/spdlog
rm -rf ~/.cache/flashinfer/
```

## Environment

Same 2-node cluster as previous tests (8× RTX PRO 6000 SM120 per node, NIXL PD, vllm-router). vLLM dev511 + FlashInfer 0.7.1 (source) + DeepGEMM 6901431 + #57292 patch. `--kernel-config '{"moe_backend":"deep_gemm"}'` (FlashInfer CUTLASS JIT unreliable on SM120).

## Results

### PD: 1K text ladder (random, ignore-eos, via router)

| Conc | Input tokens | Output tokens | tok/s | TTFT | TPOT |
|---|---|---|---|---|---|
| c1 | 8,432 | 8,192 | 178.75 | 148ms | 5.45ms |
| c4 | 33,728 | 32,768 | 439.86 | 205ms | 8.55ms |
| c8 | 67,456 | 65,536 | 597.46 | 223ms | 12.60ms |
| c16 | 134,912 | 131,072 | 840.39 | 726ms | 17.53ms |
| c32 | 101,184 | 98,304 | 1,213.74 | 418ms | 23.68ms |
| c64 | 134,912 | 131,072 | 1,766.17 | 760ms | 30.78ms |

### PD: long-context ladder (1M max_model_len)

| Tier | Input tokens | Output tokens | tok/s | TTFT | TPOT |
|---|---|---|---|---|---|
| 128K×c1 | 520,120 | 4,096 | 87.44 | 6.4s | 5.16ms |
| 128K×c2 | 1,040,240 | 8,192 | 176.30 | 3.8s | 6.59ms |
| 128K×c4 | 1,560,360 | 12,288 | 256.44 | 4.3s | 8.03ms |
| 256K×c1 | 780,090 | 3,072 | 69.71 | 7.5s | 7.05ms |
| 256K×c2 | 1,560,180 | 6,144 | 152.95 | 4.3s | 7.89ms |
| 512K×c1 | 1,040,060 | 2,048 | 37.15 | 18.2s | 9.16ms |
| 1M×c1 | 2,080,060 | 2,048 | 18.72 | 48.0s | 6.57ms |

### Single-node (for comparison)

| Conc | dev382 | dev511 |
|---|---|---|
| c1 | 187 | 149.7 |
| c4 | 261 | 343.7 |
| c8 | — | 512.5 |
| c16 | — | 744.5 |

### Agent (12-round tool-call, dev511 single-node)

12/12 success, mean tool-call round 0.29s, answer round 0.90s, prefix cache 73~90%, DSpark acceptance P0=78.6%.

## Raw Data

```
raw/d511_pd_bench/     1K ladder tier logs + progress + window
raw/d511_pd_ctx/       long-context ladder tier logs
raw/d_engine.log       decode node engine log
raw/p_engine.log       prefill node engine log
raw/router.log         vllm-router log
raw/single_node/       single-node test logs
```

## Data Boundary

This page presents only this test's own data on this hardware/software configuration. No cross-model, cross-device, or cross-framework comparisons. All metrics use official `vllm bench serve` names and units.
