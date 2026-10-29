# Qwen3.8-27B-FP8 + DFlash2（spec=7）DGX Spark(GB10) TP1 压测

测试日期 2026-09-03 ｜ vLLM nightly 0.28.1rc1.dev283 ｜ Qwen3.8-27B-FP8 ｜ TP=1 ｜ DFlash2 num_spec=7

## 结论速览
- **9/9 档全部成功**（部分矩阵：3 组 × C=1/2/4）
- 单流 decode 9.9–17.7 tok/s（TPOT 56–111 ms）；并发 4 聚合 33.0–44.0 tok/s
- DFlash2@7 接受率 13.7–30.2%，接受长度 1.96–3.11——深 7 后段 draft 大量被拒，验证开销显著
- 本版 vLLM 支持深度 7 启动（旧版只能 ≤3）
- 主模型为官方 FP8 包（KV bf16，101 万 tokens）

## 目录
- `docs/benchmark_report.html` - 完整报告
- `bench_scripts/run_spark44_dflash7_fp8.sh` - 压测脚本（含等服就绪逻辑；客户端跑独立 x86 机远程打）
- `logs/` - 9 档官方输出 JSON


## v2 扩展测试（2026-09-04 追加）

- `docs/benchmark_report_v2.html` - v2 报告（随机 token 九档 + Agent 任务流 12 轮 + 真实文本 5 组 + prefix cache 验证 + 多轮对话/热态长输出）
- `logs/v2/` - v2 九档 JSON

### v2 测试结果

| 场景 | 结果 |
|---|---|
| 单流 decode · 随机 token | 15.5 ~ 20.9 tok/s |
| 单流 decode · 真实文本 | 11.9 ~ 16.5 tok/s |
| Agent 工具调用轮 decode | 40.0 ~ 47.5 tok/s（结构化投机加速 ~3 倍） |
| Agent 12 轮 TTFT | 0.24~1.13 s（逐轮抬升） |
| 同前缀第 2 次请求 | **未命中**（15.0 s 重算），第 3 次起命中（0.78 s）——Mamba prefix cache 延迟一轮生效 |
| 8K×c4 TTFT | 22.66 s（投机调度挤压 prefill 预算） |
