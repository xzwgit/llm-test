# DeepSeek-V4.1-Flash · 8×NVIDIA B300 SXM6（SM100）TP8

测试日期 **2026-09-30** ｜ 硬件 8×B300 SXM6（275 GB/卡，SM100，驱动 595.71.05）｜ 模型 DeepSeek-V4.1-Flash ｜ **TP=8** ｜ 1M 上下文

本目录含**两组彼此独立的测试**（均只含本机本模型数据）：

| 组 | 运行环境 | 说明 |
|---|---|---|
| **A** | 官方镜像 `vllm/vllm-openai:nightly` | vLLM 0.30.1rc1.dev396 + flashinfer 0.7.0.post1 + torch 2.13.0+cu130（镜像内置，**不含 DeepGEMM**） |
| **B** | 本地 uv 环境 | vLLM 0.30.1rc1.dev412 + flashinfer 0.7.0.post1 + **DeepGEMM 6901431（源码编译）** |

## 结论速览

**A 组（官方镜像）**
- 1K×c1：**225.5 tok/s**（TTFT 55.5 ms、TPOT 4.38 ms、ITL 4.39 ms）
- 1K×c4：**793.4 tok/s**（TTFT 104.1 ms、TPOT 4.94 ms、ITL 4.97 ms）
- 零失败；**KV 缓存 185.59 GiB/卡**；首启约 26 分钟（其中 FlashInfer autotune 约 19 分钟）
- SM100 专属路径已生效：`Using MXFP4 indexer cache for Lightning Indexer`、`Using 'FLASHINFER_TRTLLM_MXFP4_MXFP8' Mxfp4 MoE backend`、`TrtLlmMxfp4ExpertsModular`
- 正确性：`17*19 → 323`

**B 组（本地 uv 环境，需 `--kernel-config '{"moe_backend":"deep_gemm"}'`）**
- 1K×c1：**146.6 tok/s**（TTFT 68.3 ms、TPOT 6.76 ms、ITL 6.80 ms）
- 1K×c4：**566.9 tok/s**（TTFT 125.1 ms、TPOT 6.93 ms、ITL 6.99 ms）
- 零失败；**KV 缓存 186.04 GiB/卡**；`DEEPGEMM_MXFP4` MoE 后端 + `DeepGEMM PDL enabled` + `DeepGEMM E8M0 enabled on current platform`
- 正确性：`17*19 → 323`

> B 组需显式指定 MoE 后端的原因：该 uv 组合（vLLM dev412 + flashinfer 0.7.0.post1）在 SM100 上默认选 `flashinfer_trtllm`，其 JIT 编译失败（`ninja: build stopped`，真错为 TRT-LLM 头文件重复定义）。改用 `deep_gemm` 后正常启动。**如实记录，不构成与其他配置的对比。**

## 测试方法与口径
- 工具：`vllm bench serve`（官方），`--backend openai-chat`，`--dataset-name random`，`--ignore-eos`，`--random-input-len 1024 --random-output-len 1024`，`--percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99`
- 每组每档前做同形状预热，避免首次形状 JIT 造成的 TTFT 假象
- 启动参数：B 组见 `bench_scripts/boot_v41_b300_uv.sh`；A 组见 `bench_scripts/docker_run_v41_image.sh`

## 指标含义（官方定义，逐列）
| 指标 | 官方含义 |
|---|---|
| Successful / Failed requests | 成功 / 失败的请求数 |
| Benchmark duration (s) | 本次测试总耗时 |
| Total input tokens | 累计输入（prompt）token 数 |
| Total generated tokens | 累计输出（生成）token 数 |
| Request throughput (req/s) | 请求吞吐 = 成功请求数 / 总耗时 |
| Output token throughput (tok/s) | 输出 token 吞吐 = 总生成 token / 总耗时 |
| Peak output token throughput (tok/s) | 官方工具按窗口统计的峰值输出吞吐（短跑时可能与均值口径不一致，如实保留官方输出） |
| TTFT（Mean/Median/P50/P95/P99，ms） | Time To First Token，首 token 延迟 |
| TPOT（Mean/Median/P50/P95/P99，ms） | Time Per Output Token，不含首 token 的每输出 token 时间 |
| ITL（Mean/Median/P50/P95/P99，ms） | Inter-Token Latency，相邻 token 间隔 |

## 目录
- `logs/` - 各组各档官方输出（`.log` 完整输出；A 组另有官方 `--save-result` 的 `.json`）
- `bench_scripts/` - 两组的启动与压测命令
