# Qwen3.8-27B 单机 TP=8 基线压测（8×RTX 5090）+ 双机对比

测试日期 2026-09-02 ｜ vLLM 0.28.0 ｜ Qwen/Qwen3.8-27B bf16 ｜ TP=8 ｜ MTP×3

## 结论速览
- **26/26 档全部成功**（口径与双机 DP=2 报告完全一致）
- 单机峰值输出吞吐 **2,337.86 tok/s**（1K×c64）；单流最高 171.3 tok/s
- **双机 DP=2 扩展比 ×1.0–×1.71**：并发 ≥4 才有收益，高并发长输出最优（8K×32K×c32 ×1.67）
- 单流低并发场景双机无收益（单流只落一台引擎）

## 目录
- `docs/benchmark_report_single.html` - 完整报告（含单机 vs 双机逐档对比表）
- `bench_scripts/run_bench_single.sh` - 压测脚本
- `logs/results-single/` - 26 档官方输出 JSON

双机数据见 `../qwen3.8-27b-dp2-16x5090/`
