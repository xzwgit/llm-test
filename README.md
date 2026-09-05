# llm-test

LLM 部署与性能测试归档。每个子目录是一个独立测试项目，报告只含本测试自身数据（保真，不含跨模型/跨设备对比）。

## 模型索引

### DeepSeek-V4-Pro
| 测试 | 日期 | 说明 |
|---|---|---|
| [deepseek-v4-pro](deepseek-v4-pro/) | 2026-08 | DP2+MTP 部署, 多上下文压测, SM120 补丁与调优 |

### Qwen3.8-27B
| 测试 | 日期 | 说明 |
|---|---|---|
| [qwen3.8-27b-dp2-16x5090](qwen3.8-27b-dp2-16x5090/) | 2026-09 | 双机 DP=2（16×RTX 5090）26 档 |
| [qwen3.8-27b-tp8-single-8x5090](qwen3.8-27b-tp8-single-8x5090/) | 2026-09 | 单机 TP=8（8×RTX 5090）26 档 |
| [qwen3.8-27b-nvfp4-tp1-gb10-spark](qwen3.8-27b-nvfp4-tp1-gb10-spark/) | 2026-09 | unsloth NVFP4 包, DGX Spark(GB10) 9 档部分矩阵 |
| [qwen3.8-27b-fp8-tp1-gb10-spark](qwen3.8-27b-fp8-tp1-gb10-spark/) | 2026-09 | 官方 FP8 包（KV bf16）, DGX Spark(GB10) 9 档 |
| [qwen3.8-27b-dflash2-tp4-4x4090](qwen3.8-27b-dflash2-tp4-4x4090/) | 2026-08 | DFlash2(spec=3) 投机解码, 4×RTX 4090 TP4, 21 档（docker nightly） |
| [qwen3.8-27b-fp8-dflash2-tp1-gb10-spark](qwen3.8-27b-fp8-dflash2-tp1-gb10-spark/) | 2026-09 | FP8 主模型 + DFlash2(spec=7), DGX Spark(GB10) 九档 + v2（Agent/多轮/缓存验证） |
| [qwen3.8-27b-fp8-dflash2-tp8-8x5090](qwen3.8-27b-fp8-dflash2-tp8-8x5090/) | 2026-09 | FP8 + DFlash2(spec=7), 8×RTX 5090 TP8, 26 档矩阵 14 档有效（含两次崩溃记录） |
| [qwen3.8-27b-sglang-dspark-tp1-gb10-spark](qwen3.8-27b-sglang-dspark-tp1-gb10-spark/) | 2026-09 | SGLang 定制镜像 + RadixArk NVFP4 + DSPARK, DGX Spark(GB10) 9 档 + 场景化全套 + 无投机对照 |

### Qwen3.6-27B
| 测试 | 日期 | 说明 |
|---|---|---|
| [qwen3.6-27b-nvfp4-tp1-gb10-spark](qwen3.6-27b-nvfp4-tp1-gb10-spark/) | 2026-09 | 官方 NVFP4 包, DGX Spark(GB10) 9 档部分矩阵 |

## 目录约定

```
<项目目录>/
├── README.md          # 本项目结论速览 + 范围 + 启动命令
├── docs/              # benchmark_report.html 完整报告
├── bench_scripts/     # 压测脚本
└── logs/              # 官方输出 JSON（逐档）
```

## 如何新增测试

1. 复制 `_template/` 为新目录（命名: 模型-量化/加速-TP-硬件）
2. 子目录内放 `docs/`, `bench_scripts/`, `logs/`
3. 在本 README「模型索引」对应模型分组下登记一行
4. git commit + push（默认分支 master）

## 环境要求

见各子项目 docs/；压测客户端一律跑在独立服务器上远程打被测机。
