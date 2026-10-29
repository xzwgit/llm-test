# DeepSeek-V4-Pro 调优参数建议

> 基于 2×8 RTX PRO 6000 (96GB, SM120) + DP2 + MTP 实测得出

## max-num-batched-tokens (prefill 批量)

**含义**: 单次 forward (前向推理) 能处理的最大 token 数。与时间无关, 不是"每秒"。
vLLM 推理循环每个 step 打包 ≤此值的 token 送 GPU 计算。

**主要影响 prefill (输入)**, 不影响 decode (输出):
- prefill: 一次吃 N 个 token, 超了就分块
- decode: 每 seq 每 step 只 1 token, 占用极小

**必须有上限**: 长输入一次 prefill 会爆显存 (激活峰值)。

### 默认值 vs 推荐值

- **默认**: 2048 (vLLM 0.26 SchedulerConfig)
- **推荐**: `--max-num-batched-tokens 8192`

### 各输入长度的 prefill 次数 (实测 token 数)

实测 prompt token 数 (DeepSeek-V4-Pro tokenizer):
| 脚本 | prompt token | @2048(默认) | @8192(推荐) |
|---|---|---|---|
| 4K输入 (×12) | 673 | 1 次 | 1 次 (无变化) |
| 16K输入 (×125) | 7001 | **4 次** | **1 次** |
| 32K输入 (×255) | 14281 | **7 次** | **2 次** |
| 32K32k (×2400) | 24001 | 12 次 | 3 次 |

**结论**:
- 短输入 (≤4K, <2048 token): 调不调无区别 (都 1 次 prefill)
- 16K 输入: 4 次 → 1 次, TTFT 明显改善
- 32K 输入: 7 次 → 2 次, TTFT 减半左右

### 选择依据

| 值 | 适用 | 风险 |
|---|---|---|
| 2048 (默认) | 仅短输入 (≤4K) | 长输入 TTFT 偏高 |
| **8192 (推荐)** | 通用, 长输入 TTFT 改善 | 激活峰值可控 (数 GB) |
| 16384 (激进) | 追求极致 TTFT | 显存余量紧张时可能 prefill OOM |

### 注意

- 当前显存余量 ~1GB/卡 (gpu-memory-utilization 0.95), 激进值 (16384) 有 OOM 风险
- 此参数只影响 prefill 分块, 不限制输出长度 (输出受 max_tokens / max-model-len 管)
- 改此参数需重启 vllm (运行中改不了)

## 其他参数 (当前配置, 已验证)

| 参数 | 值 | 说明 |
|---|---|---|
| tensor-parallel-size | 8 | 节点内, 不能更大 (单机 8 卡) |
| data-parallel-size | 2 | 跨节点, 两台机各一副本 |
| enable-expert-parallel | (开) | EP8, 每卡 48 专家 |
| moe-backend | marlin | DeepGEMM+patch 也可 (默认 auto=DeepGEMM) |
| kv-cache-dtype | fp8 | |
| block-size | 256 | |
| gpu-memory-utilization | 0.95 | 每卡占满 ~97GB |
| max-model-len | 524288 | 512K (1M+MTP 会 OOM) |
| speculative-config | mtp, num_spec=2 | 投机解码 |
| max-num-seqs | (默认 128) | 调度并发上限, 一般够用 |

## 并发与 OOM 关系

- **增加并发请求**: 不会 OOM (vLLM 抢占式调度, KV 池满了就排队/换出), 只会变慢
- **长上下文 (接近 512K) + 高并发**: prefill 激活峰值可能突破显存余量, 有 OOM 风险
- **KV cache 容量**: MLA 压缩 (576 B/token), 8 卡池子 ~304GB, 能容纳 ~1000 个满上下文请求

## 显存分解 (每卡 96GB)

| 项目 | 占用 |
|---|---|
| 权重 (FP4专家48个 + FP8共享TP8切分) | ~54 GB |
| KV cache 池 | ~38 GB |
| 激活 + CUDA graph | ~5 GB |
| 余量 | ~1 GB |
| 合计 | ~97 GB |

## 启动脚本

- server01 (主): `scripts/start_server01.sh`
- server02 (headless): `scripts/start_server02.sh`

两脚本已包含 `--max-num-batched-tokens 8192`, 直接 `bash scripts/start_serverXX.sh` 启动。
