# Qwen3.8-27B-NVFP4（unsloth）单机 TP=1 压测（DGX Spark GB10）

测试日期 2026-09-02 ｜ vLLM nightly 0.28.1rc1.dev283 ｜ unsloth/Qwen3.8-27B-NVFP4 ｜ TP=1 ｜ MTP×3

## 结论速览
- **9/9 档全部成功**（部分矩阵：3 组 × C=1/2/4）
- 单流 decode 17–24 tok/s（TPOT 41–58 ms）；并发 4 聚合 58.6–76.4 tok/s
- MTP 接受率 38.7–70.8%，接受长度 2.16–3.12
- kernel 路径：FlashInfer CuteDSL NVFP4（SM121 无原生 FP4 单元，weight-only）
- 连带特征：KV 池 50.33 GiB（1,420,556 tokens）；首次启动 autotune ~9 分钟
  （需 `VLLM_ENGINE_READY_TIMEOUT_S=3600`，结果进 cache 后恢复正常）

## 范围说明（部分矩阵）
- 已测：1K→1K、4K→4K、8K→8K × 并发 1/2/4
- 未测：4K→32K、8K→32K 两组及 16/32/64 高并发档

## 目录
- `docs/benchmark_report.html` - 完整报告（含指标说明）
- `bench_scripts/run_spark44_qwen38.sh` - 压测脚本（客户端跑在独立 x86 服务器，远程打被测机）
- `logs/` - 9 档官方输出 JSON
