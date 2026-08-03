# DeepSeek-V4-Pro 调优参数建议

> 基于 2×8 RTX PRO 6000 (96GB, SM120) + DP2 + MTP 实测得出

## 关键参数

### max-num-batched-tokens (prefill 批量)

**作用**: 每次 forward 最多处理的 token 数 (prefill + decode 合计)。
影响长输入的 prefill 分块次数和 TTFT。

**默认值**: 2048 (vLLM 0.26 SchedulerConfig 默认)

**推荐值**: `--max-num-batched-tokens 8192`

#### 不同值对长输入的影响

| max-num-batched-tokens | 32K输入 prefill 次数 | 预期 TTFT |
|---|---|---|
| 2048 (默认) | 7 次 | ~4.1s |
| **8192 (推荐)** | **2 次** | **~1.5s** |
| 16384 (激进) | 1 次 | ~0.8s |

#### 选择依据

- **8192 (推荐)**: 长输入 TTFT 减半, 激活内存峰值可控 (数 GB 内)
- **16384 (激进)**: TTFT 最优, 但激活内存峰值高, 显存余量紧张时可能 prefill OOM
- **2048 (默认)**: 短输入 (≤4K) 够用, 不用改; 长输入 TTFT 偏高

#### 注意

- 显存余量紧张时 (gpu-memory-utilization 0.95 下每卡仅剩 ~1GB), 激进值 (16384) 有 prefill OOM 风险
- 短输入场景 (≤4K) 调此参数收益不大 (2048 已能一次 prefill 完)
- 此参数只影响 prefill 分块, 不影响 decode 并发上限 (decode 每 step 每 seq 1 token)

### 其他参数 (当前配置, 已验证)

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
