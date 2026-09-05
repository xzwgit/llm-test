# Qwen3.8-27B-FP8 + DFlash2（spec=7）· 8×RTX 5090 TP8 压测

测试日期 2026-09-05 ｜ vLLM 0.28.0 ｜ Qwen3.8-27B-FP8 ｜ TP=8（PCIe 无 NVLink）｜ DFlash2 num_spec=7

## 结论速览
- 26 档矩阵中 14 档有效（1K 全组/4K 全组/8K c1-c4），其余因两次服务崩溃未获得数据（报告第 5 节如实记录）
- 单流 decode 15.5~20.9 tok/s（TPOT 46.6~64.0 ms）
- 峰值聚合 97.5 tok/s（1K×c64），但 TPOT 恶化至 480 ms、TTFT 88.5 s
- DFlash2 接受率 15.4~27.9%、接受长 2.08~2.96（Drafted≈4×Accepted）
- 两次崩溃均在 DFlash2 投机路径高并发段（speculator OOM / worker RPC 超时）

## 目录
- `docs/benchmark_report.html` - 完整报告（15 列 + 4.4 投机逐档明细 + 崩溃记录）
- `logs/` - 26 档官方输出 JSON
- `bench_scripts/run_45_dflash7.sh` - 压测编排
