# Qwen3.6-27B-NVFP4 单机 TP=1 压测（DGX Spark GB10）

测试日期 2026-09-02 ｜ vLLM nightly 0.28.1rc1.dev283 ｜ nvidia/Qwen3.6-27B-NVFP4 ｜ TP=1 ｜ MTP×3

## 结论速览
- **9/9 档全部成功**（部分矩阵：3 组 × C=1/2/4）
- 单流 decode **26–30 tok/s**（TPOT 32–37 ms）；C=4 聚合 78–86 tok/s
- MTP 接受率 63.5–85.9%（档均值），单请求稳态 96.9%、接受长度 3.91/4.0
- prefill 是短板：8K 输入 C=4 P95 TTFT 22.66 s
- NVFP4 走 Marlin weight-only kernel（SM121 无原生 FP4 单元）

## 范围说明（部分矩阵）
- 已测：1K→1K、4K→4K、8K→8K × 并发 1/2/4
- 未测：4K→32K、8K→32K 两组及 16/32/64 高并发档

## 目录
- `docs/benchmark_report.html` - 完整报告（含指标说明表）
- `bench_scripts/run_spark44.sh` - 压测脚本（客户端跑在独立 x86 服务器，远程打被测机）
- `logs/` - 9 档官方输出 JSON

## 环境要点
- GB10（SM121）121G 统一内存；gpu-memory-utilization 0.95 为上限（0.96/0.98 无法启动）
- KV cache fp8_e4m3 为 NVFP4 checkpoint 自带；KV 容量 1,833,543 tokens（256K×6.99 路）
- prefix caching 被 MTP+Mamba 架构自动禁用
