# Qwen3.8-27B-FP8（官方包）单机 TP=1 压测（DGX Spark GB10）+ 三方对照

测试日期 2026-09-02 ｜ vLLM nightly 0.28.1rc1.dev283 ｜ Qwen3.8-27B-FP8 ｜ TP=1 ｜ MTP×3

## 结论速览
- **9/9 档全部成功**（部分矩阵：3 组 × C=1/2/4）
- **意外结论：官方 FP8 包是三方案中最慢**（单流 14–17 tok/s，TPOT 58–78 ms）
- 三方排序（同机同口径实测）：**nvidia NVFP4 (Marlin) > unsloth NVFP4 (CuteDSL) > 官方 FP8 (Cutlass)**

## 根因
- 该 FP8 checkpoint 的 **KV cache 为 bf16（未量化）**：KV 容量仅 101 万 token（三方最小），
  decode 每 step 读 KV 数据量是 fp8 KV 的 2 倍 → TPOT 三方最差
- GEMM 走 CutlassFp8BlockScaled（SM121 上效率一般，"Not enough SMs for max_autotune_gemm"）
- 修正：issue #186 的 "fp8 fits very well" 指 DeepSeek 栈（量化 KV），不适用于 Qwen 官方 FP8 包

## 范围说明（部分矩阵）
- 已测：1K→1K、4K→4K、8K→8K × 并发 1/2/4
- 未测：4K→32K、8K→32K 两组及 16/32/64 高并发档

## 目录
- `docs/benchmark_report.html` - 完整报告（含三方对照表）
- `bench_scripts/run_spark44_fp8.sh` - 压测脚本（含等服就绪逻辑；客户端跑独立 x86 机远程打）
- `logs/` - 9 档官方输出 JSON

对照数据：`../qwen3.6-27b-nvfp4-tp1-gb10-spark/`、`../qwen3.8-27b-nvfp4-tp1-gb10-spark/`
待验证：FP8 + `--kv-cache-dtype fp8`（KV 带宽减半，预期 TPOT 改善）
