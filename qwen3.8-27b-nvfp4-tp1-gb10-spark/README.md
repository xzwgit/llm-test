# Qwen3.8-27B-NVFP4（unsloth）单机 TP=1 压测（DGX Spark GB10）+ 同机官方包对照

测试日期 2026-09-02 ｜ vLLM nightly 0.28.1rc1.dev283 ｜ unsloth/Qwen3.8-27B-NVFP4 ｜ TP=1 ｜ MTP×3

## 结论速览
- **9/9 档全部成功**（部分矩阵：3 组 × C=1/2/4）
- 单流 decode 17–24 tok/s（TPOT 41–58 ms）
- **对比同机 nvidia/Qwen3.6-27B-NVFP4（Marlin）：吞吐 -10~35%、TPOT 恶化 15~55%**
- 连带代价：KV 池 50.33 GiB（官方包路线 64.98，-22%）；首次启动 autotune ~9 分钟（需 `VLLM_ENGINE_READY_TIMEOUT_S=3600`）
- 外部佐证：MiaAI-Lab issue #186（2026-09-02）"fp8 fits very well and much higher quality"，不建议当前在 GB10 用 NVFP4

## 根因
SM121 无原生 FP4 计算单元；unsloth 包（compressed-tensors）走 FlashInfer CuteDSL kernel，
官方包（modelopt）走 Marlin——前者在该平台既慢又吃显存。

## 范围说明（部分矩阵）
- 已测：1K→1K、4K→4K、8K→8K × 并发 1/2/4
- 未测：4K→32K、8K→32K 两组及 16/32/64 高并发档

## 目录
- `docs/benchmark_report.html` - 完整报告（含同机对照表与指标说明）
- `bench_scripts/run_spark44_qwen38.sh` - 压测脚本（客户端跑在独立 x86 服务器，远程打被测机）
- `logs/` - 9 档官方输出 JSON

对照数据（同机 Qwen3.6-NVFP4）：`../qwen3.6-27b-nvfp4-tp1-gb10-spark/`
