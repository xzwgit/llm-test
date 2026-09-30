# DeepSeek-V4-Flash-0731 · 8×NVIDIA B300 SXM6（SM100）TP8

测试日期 **2026-09-30** ｜ 硬件 8×B300 SXM6（275 GB/卡，SM100，驱动 595.71.05）｜ 模型 DeepSeek-V4-Flash-0731（`model_type: deepseek_v4`，43 层 / hidden 4096 / 256 专家 / 无 Engram）｜ **TP=8** ｜ 运行环境 **官方镜像 `vllm/vllm-openai:nightly`**（vLLM 0.30.1rc1.dev396 + flashinfer 0.7.0.post1，不含 DeepGEMM）

## 结论速览
- 正确性：`17*19 → 323`
- **KV 缓存 212.18 GiB/卡**；首启约 21 分钟（含 FlashInfer autotune）
- 投机解码：`--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'`（V4-Flash 官方 recipe 配置），日志 `DSpark draft model loaded: 99 params`；MoE 后端 `FLASHINFER_TRTLLM_MXFP4_MXFP8`
- 零失败（两档两轮共 4 次测量）

**热态（预热后复测，推荐引用这一组）**
| 档位 | 输出 tok/s | TTFT 均值 | TPOT 均值 | ITL 均值 |
|---|---|---|---|---|
| 1K×c1 | **448.2** | 55.9 ms | 2.18 ms | 7.05 ms |
| 1K×c4 | **938.9** | 255.6 ms | 2.93 ms | 7.97 ms |

**冷态（服务就绪后首测，作为冷启动参考一并保留）**
| 档位 | 输出 tok/s | TTFT 均值 | TPOT 均值 |
|---|---|---|---|
| 1K×c1 | 110.9 | **7136.7 ms** | 2.05 ms |
| 1K×c4 | 297.7 | **9403.1 ms** | 3.24 ms |

> **冷启动说明（重要）**：冷态那两档的 TTFT 高达 7.1 s / 9.4 s，是**首次遇到该输入形状时的 JIT/编译开销**；同形状预热后 TTFT 降至 55.9 ms / 255.6 ms。两组为**同一配置**的先后测量，热态组是稳态表现。

## 测试方法与口径
- 工具：`vllm bench serve`（官方），`--backend openai-chat`，`--dataset-name random`，`--ignore-eos`，`--random-input-len 1024 --random-output-len 1024`，`--percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99`
- 启动参数见 `bench_scripts/docker_run_v4flash_image.sh`

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
- `logs/` - 官方输出（`*-cold.log` / `*-warm.log` 完整输出；c1/c4 另有官方 `--save-result` 的 `.json`）
- `bench_scripts/` - 启动与压测命令
