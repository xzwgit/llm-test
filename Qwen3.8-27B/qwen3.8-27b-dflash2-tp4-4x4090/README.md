# Qwen3.8-27B-FP8 + DFlash2（spec=3）4×RTX 4090 TP4 压测

测试日期 2026-08-26 ｜ docker vllm/vllm-openai:nightly（0.26.1rc1.dev1177）｜ Qwen3.8-27B-FP8 ｜ TP=4（GPU 4-7）｜ DFlash2 num_spec=3

## 结论速览
- **21/21 档全部成功**（1K/4K/8K/4K→32K 四组 × C=1/2/4/16/32，1K 组加 64；8K→32K 未测）
- 单流 decode 79–117 tok/s（TPOT 8–12 ms）；峰值聚合 1,064.90 tok/s（1K×C64）
- DFlash2 接受率 28.9–61.0%，接受长度 1.87–2.83（spec=3）
- 权重走 Marlin kernel（SM8.9 无原生 FP8）；KV 约 152 万 tokens

## 目录
- `docs/benchmark_report.html` - 完整报告（含指标说明）
- `logs/` - 21 档官方输出 JSON（当次测试在 docker 容器内起服务，宿主机 bench 远程打）
