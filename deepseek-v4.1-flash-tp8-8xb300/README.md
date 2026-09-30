# DeepSeek-V4.1-Flash · 8× NVIDIA B300 SXM6 (SM100), TP8

Test date **2026-09-30** ｜ Hardware: 8× B300 SXM6 (275 GB/GPU, SM100, driver 595.71.05) ｜ Model: DeepSeek-V4.1-Flash ｜ **TP=8** ｜ 1M context

> 中文版见 [README.zh.md](README.zh.md)

This directory contains **two independent test groups** (each with its own configuration; no cross-config comparison is made):

| Group | Runtime | Notes |
|---|---|---|
| **A** | Official image `vllm/vllm-openai:nightly` | vLLM 0.30.1rc1.dev396 + flashinfer 0.7.0.post1 + torch 2.13.0+cu130 (bundled in the image, **no DeepGEMM**) |
| **B** | Local uv environment | vLLM 0.30.1rc1.dev412 + flashinfer 0.7.0.post1 + **DeepGEMM 6901431 (built from source)** |

## Results

**Group A (official image)**
- 1K×c1: **225.5 tok/s** (TTFT 55.5 ms, TPOT 4.38 ms, ITL 4.39 ms)
- 1K×c4: **793.4 tok/s** (TTFT 104.1 ms, TPOT 4.94 ms, ITL 4.97 ms)
- Zero failed requests; **KV cache 185.59 GiB/GPU**; first boot ≈ 26 min (of which ≈ 19 min is the first-time FlashInfer autotune)
- SM100-only paths active: `Using MXFP4 indexer cache for Lightning Indexer`, `Using 'FLASHINFER_TRTLLM_MXFP4_MXFP8' Mxfp4 MoE backend`, `TrtLlmMxfp4ExpertsModular`
- Correctness: `17*19 → 323`

**Group B (local uv environment; requires `--kernel-config '{"moe_backend":"deep_gemm"}'`)**
- 1K×c1: **146.6 tok/s** (TTFT 68.3 ms, TPOT 6.76 ms, ITL 6.80 ms)
- 1K×c4: **566.9 tok/s** (TTFT 125.1 ms, TPOT 6.93 ms, ITL 6.99 ms)
- Zero failed requests; **KV cache 186.04 GiB/GPU**; `DEEPGEMM_MXFP4` MoE backend + `DeepGEMM PDL enabled` + `DeepGEMM E8M0 enabled on current platform`
- Correctness: `17*19 → 323`

> Why Group B needs an explicit MoE backend: with that uv combination (vLLM dev412 + flashinfer 0.7.0.post1) the default choice on SM100 is `flashinfer_trtllm`, whose JIT compilation fails here (`ninja: build stopped`; the underlying error is a duplicate TRT-LLM header definition). Selecting `deep_gemm` starts normally. Recorded as-is; this is not a comparison against any other configuration.

## Method and measurement conventions
- Tool: `vllm bench serve` (official), `--backend openai-chat`, `--dataset-name random`, `--ignore-eos`, `--random-input-len 1024 --random-output-len 1024`, `--percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99`
- Each tier is preceded by a same-shape warm-up so that first-shape JIT does not appear as a TTFT artefact
- Startup commands: Group B in `bench_scripts/boot_v41_b300_uv.sh`; Group A in `bench_scripts/docker_run_v41_image.sh`

## Metric definitions (official, per column)
| Metric | Official meaning |
|---|---|
| Successful / Failed requests | Number of successful / failed requests |
| Benchmark duration (s) | Total wall time of the run |
| Total input tokens | Cumulative prompt tokens |
| Total generated tokens | Cumulative generated tokens |
| Request throughput (req/s) | Successful requests / duration |
| Output token throughput (tok/s) | Total generated tokens / duration |
| Peak output token throughput (tok/s) | Windowed peak reported by the official tool (on very short runs this windowed figure can differ from the mean; the official output is reported as-is) |
| TTFT (Mean/Median/P50/P95/P99, ms) | Time To First Token |
| TPOT (Mean/Median/P50/P95/P99, ms) | Time Per Output Token, excluding the first token |
| ITL (Mean/Median/P50/P95/P99, ms) | Inter-Token Latency |

## Contents
- `logs/` — official output per tier (`.log` full output; Group A also has the official `--save-result` `.json`)
- `bench_scripts/` — startup and benchmark commands for both groups
