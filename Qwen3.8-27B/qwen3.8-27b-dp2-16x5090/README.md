# Qwen3.8-27B 双机 DP=2 压测（16×RTX 5090）

测试日期 2026-09-01 ｜ vLLM 0.28.0 ｜ Qwen/Qwen3.8-27B bf16 ｜ TP=8/机 + DP=2 ｜ MTP×3

## 结论速览
- **26/26 档全部成功**（5 组输入/输出组合 × 并发 1/2/4/16/32，1K 组加测 64）
- 峰值输出吞吐 **4,005 tok/s**（1K 输入×1K 输出×并发 64）
- 32K 长输出并发 32 保持 1,727–1,800 tok/s；单流最高 118 tok/s
- MTP 投机接受率 18.6–82.5%（官方 spec_decode_acceptance_rate 字段）
- 跨机 NCCL/GPUDirect RDMA 验证通过（DP 模式跨机无集合通信；详见报告第 3 节）

## 目录
- `docs/benchmark_report.html` - 完整报告（指标含义、部署配置、NCCL 验证、26 档全数据）
- `docs/benchmark_summary.md` - 数据汇总表
- `docs/deploy_config.md` - 部署与启动配置
- `bench_scripts/run_bench.sh` - 26 档压测脚本
- `logs/results/` - 26 档官方输出 JSON 原始数据

## 硬件
2 节点 × 8× RTX 5090 32G；CX7 200G×2 直连（RoCE v2 + GPUDirect RDMA，无交换机）。
