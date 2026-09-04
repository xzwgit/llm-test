# Qwen3.8-27B · SGLang 定制镜像 + DSPARK · 1×GB10 (DGX Spark) TP1

- 日期: 2026-09-03/04
- 框架: SGLang 0.0.0.dev0+qwen38.27b.g561c8f3（容器 lmsysorg/sglang:qwen38-27b，CUDA 13.0.3）
- 主模型: RadixArk/Qwen3.8-27B-NVFP4（modelopt_mixed，权重 22.17GB）
- 投机解码: DSPARK（请求 num-steps=5 被镜像覆盖为 1，实际 num_steps=1 + num_draft_tokens=8）
- KV cache: fp8_e4m3 · attention: flashinfer · context 256K · max-running-requests 8

## 结论速览

| 场景 | 结果 |
|---|---|
| 单流 decode · 随机 token | 21.6 ~ 35.8 tok/s（TPOT 27.6~46.0 ms） |
| 单流 decode · 真实文本 | 16.9 ~ 28.6 tok/s |
| 单流 decode · Agent 工具调用 JSON | 52.2 ~ 69.0 tok/s（峰值 69.0） |
| 并发 4 聚合 | 45.9 ~ 69.1 tok/s（8K×c4 最高 69.1） |
| Agent 12 轮 TTFT | 0.18~0.25 s（prefix cache 全命中） |
| 同前缀第 2 次请求 TTFT | 0.24 s（13.8K 前缀立即命中，27 倍加速） |
| 无投机对照（Agent 工具轮） | ~12.8 tok/s → 投机净收益 4~5 倍 |

## 测试范围

- 随机 token 九档（1K/4K/8K × c1/2/4，vllm bench serve 官方口径）
- 真实文本单流 5 组（中/英/代码 × 开/关思考）
- Prefix cache 验证 3 次、多轮对话 5 轮、热态长输出 3 组
- Agent 任务流 12 轮（system 工具定义 + 逐轮追加工具结果）
- 关闭投机解码对照（同一 Agent 脚本 12 轮）

## 启动命令

见 docs/benchmark_report.html 第 2 节（docker run 完整命令）。

## 目录

- docs/benchmark_report.html: 完整报告
- logs/: 九档官方输出 JSON
- bench_scripts/: 压测与场景化脚本（run_spark44_sglang.sh / supplement_b_agent.py 等）

## 备注

- DSPARK 投机在任何文本类型上均为正贡献（结构化 4~5x、自然语言 1.3~2.7x、随机 1.7~2.9x）
- Mamba/GDN 状态随 radix cache 复用（行为细节见报告第 6 节）
