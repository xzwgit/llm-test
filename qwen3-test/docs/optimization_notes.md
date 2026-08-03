# Qwen3.6-27B 脚本分析与优化建议

## 一、压测脚本 (qwen_benchmark.sh) 问题

### 问题1: 样本量太少
- 原版: `--num-prompts $((i*2))` → C=1 只测 2 个请求
- 改进: 固定 30 个 (v2脚本已改), 统计才稳定

### 问题2: 轮间间隔太短
- 原版: sleep 5 → 上一轮 KV cache 没清完
- 改进: sleep 15 (v2脚本已改)

### 问题3: 缺少 warmup
- 原版: 第一轮直接算 (含 JIT 预热, 偏慢)
- 改进: 加 --warmup 首轮不计 (v2脚本已加)

### 问题4: 未提取 TTFT
- 改进: v2 自动提取 TTFT/吞吐/延迟到 summary.csv

## 二、启动脚本 (manager) 问题

### 🔴 Bug: 前后台 Rust frontend 不一致
- 前台 line 161: `VLLM_USE_RUST_FRONTEND=0`
- 后台 line 218: `VLLM_USE_RUST_FRONTEND=1`
- **影响**: 同一模型两种模式行为不同
- **修正**: 统一为 0 (qwen3_start.sh 已修正)

### 🟡 单卡跑 27B 显存浪费
- 27B-FP8 权重 ~27GB, 单卡 96GB 只用 28%
- 若想充分利用 8 卡, 三种方案:

| 方案 | 配置 | 适用 |
|---|---|---|
| DP8 (最猛) | TP1, 8个副本 | 纯吞吐优先 |
| TP4+DP2 | TP4, 2副本 | 平衡 |
| TP8 | TP8 | 单请求最快 |

### 🟡 MAX_MODEL_LEN=262144 偏激进
- 27B 单卡 256K 上下文, KV cache 池会被压缩
- 建议: 降到 131072 (128K), 更稳

### 🟡 gpu-memory-utilization=0.95 偏高
- 长输出时激活峰值可能 OOM
- 建议: 0.92 (留 8GB 余量)

### 🟢 --mm-encoder-tp-mode data 可能无效
- 这是多模态参数, 若 Qwen3.6-27B 是纯文本模型则无效
- 确认模型是否多模态, 若否则删除

## 三、v2 脚本改进对照

| 项 | 原版 | v2 |
|---|---|---|
| 样本量 | i*2 (太少) | 30 (固定) |
| 轮间间隔 | 5s | 15s |
| warmup | 无 | --warmup 选项 |
| TTFT提取 | 无 | summary.csv |
| 模型可配 | 写死 | --model 参数 |
