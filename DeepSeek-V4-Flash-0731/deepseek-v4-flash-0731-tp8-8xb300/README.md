# DeepSeek-V4-Flash-0731 · 8× NVIDIA B300 SXM6 (SM100), TP8

Test date **2026-09-30** ｜ Hardware: 8× B300 SXM6 (275 GB/GPU, SM100, driver 595.71.05) ｜ Model: DeepSeek-V4-Flash-0731 (`model_type: deepseek_v4`, 43 layers / hidden 4096 / 256 experts / no Engram) ｜ **TP=8** ｜ Runtime: **official image `vllm/vllm-openai:nightly`** (vLLM 0.30.1rc1.dev396 + flashinfer 0.7.0.post1, no DeepGEMM)

> 中文版见 [README.zh.md](README.zh.md)

## Results
- Correctness: `17*19 → 323`
- **KV cache 212.18 GiB/GPU**; first boot ≈ 21 min (includes the first-time FlashInfer autotune)
- Speculative decoding: `--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'` (the V4-Flash recipe configuration); log shows `DSpark draft model loaded: 99 params`; MoE backend `FLASHINFER_TRTLLM_MXFP4_MXFP8`
- Zero failed requests (4 measurements across two tiers and two rounds)

**Warm (re-measured after warm-up — this is the recommended set to quote)**
| Tier | Output tok/s | TTFT mean | TPOT mean | ITL mean |
|---|---|---|---|---|
| 1K×c1 | **448.2** | 55.9 ms | 2.18 ms | 7.05 ms |
| 1K×c4 | **938.9** | 255.6 ms | 2.93 ms | 7.97 ms |

**Cold (first run right after the server became ready — kept as a cold-start reference)**
| Tier | Output tok/s | TTFT mean | TPOT mean |
|---|---|---|---|
| 1K×c1 | 110.9 | **7136.7 ms** | 2.05 ms |
| 1K×c4 | 297.7 | **9403.1 ms** | 3.24 ms |

> **Cold-start note (important)**: the cold TTFT of 7.1 s / 9.4 s is **first-shape JIT/compilation overhead**; after a same-shape warm-up TTFT drops to 55.9 ms / 255.6 ms. The two groups are successive measurements of the **same configuration**; the warm set is the steady-state behaviour.

## Method and measurement conventions
- Tool: `vllm bench serve` (official), `--backend openai-chat`, `--dataset-name random`, `--ignore-eos`, `--random-input-len 1024 --random-output-len 1024`, `--percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99`
- Startup command: `bench_scripts/docker_run_v4flash_image.sh`

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
- `logs/` — official output (`*-cold.log` / `*-warm.log` full output; `*.json` from the official `--save-result` for c1/c4)
- `bench_scripts/` — startup and benchmark commands
