# Qwen3.8-27B-FP8（官方包）单机 TP=1 压测（DGX Spark GB10）

测试日期 2026-09-02 ｜ vLLM nightly 0.28.1rc1.dev283 ｜ Qwen3.8-27B-FP8（本地权重目录） ｜ TP=1 ｜ MTP×3

## 结论速览
- **9/9 档全部成功**（部分矩阵：3 组 × C=1/2/4）
- 单流 decode 14–17 tok/s（TPOT 58–78 ms）；并发 4 聚合 47.1–52.0 tok/s
- MTP 接受率 45.4–76.8%，接受长度 2.36–3.30
- **关键发现：该官方 FP8 checkpoint 的 KV cache 为 bf16（未量化）**——
  KV 容量 1,014,183 tokens（256K 满长并发约 3.87 路），decode 每 step 读 KV
  数据量为量化 KV 的 2 倍，带宽密集的 decode 直接受累
- GEMM 走 CutlassFp8BlockScaled（"Not enough SMs for max_autotune_gemm"）

## 范围说明（部分矩阵）
- 已测：1K→1K、4K→4K、8K→8K × 并发 1/2/4
- 未测：4K→32K、8K→32K 两组及 16/32/64 高并发档
- 待验证：`--kv-cache-dtype fp8`（KV 量化）

## 目录
- `docs/benchmark_report.html` - 完整报告（含指标说明）
- `bench_scripts/run_spark44_fp8.sh` - 压测脚本（含等服就绪逻辑；客户端跑独立 x86 机远程打）
- `logs/` - 9 档官方输出 JSON
