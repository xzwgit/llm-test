# DeepSeek-V4-Pro 部署与压测记录

## 硬件环境
- 2× 服务器: 10.10.3.15 (server01) / 10.10.3.16 (server02)
- 每台 8× RTX PRO 6000 Blackwell (96GB, CC 12.0/SM120)
- 8× ConnectX-8 IB (400G each), 无 NVLink (PCIe P2P)
- CUDA 13.0, vLLM 0.26.0, NCCL 2.28.9

## 模型
- DeepSeek-V4-Pro: 861.61B 参数, 864.74GB (FP4+FP8 混合)
- 384 路由专家 (FP4/MXFP4), 6 专家/token
- MLA 注意力 (KV cache 极小: 576 bytes/token)

## 关键 Patch (SM120 必需)
SM120 上 DeepGEMM einsum 的 SM100 packed scale 布局会产生 NaN 导致乱码。
修复: patch/patch_sm120.py (改 o_proj.py + fp8_utils.py)
对应上游 PR: https://github.com/vllm-project/vllm/pull/48052

## 最终启动配置 (DP2 + MTP, 512K上下文)
拓扑: 每台机一个 TP8+EP8 副本, 两台组成 DP2
- tensor-parallel-size 8
- data-parallel-size 2 (每台 local 1)
- enable-expert-parallel
- moe-backend marlin
- kv-cache-dtype fp8
- speculative mtp num_speculative_tokens=2
- max-model-len 524288 (512K, 1M会OOM因MTP sampler buffer)
- gpu-memory-utilization 0.95
- VLLM_USE_RUST_FRONTEND=1

## 显存分解 (每卡 96GB)
| 项目 | 占用 |
|---|---|
| 权重 (FP4专家48个 + FP8共享TP8切分) | ~54 GB |
| KV cache 池 | ~38 GB |
| 激活 + CUDA graph | ~5 GB |
| 合计 | ~97 GB (占满) |

## 压测结果

### 4K/4K (max_model_len=512K)
| 并发 | TTFT | 总吞吐 | 总延迟 | 成功率 |
|---|---|---|---|---|
| 1 | 0.59s | 74.7 | 47.0s | 1/1 |
| 2 | 0.69s | 114.9 | 48.8s | 2/2 |
| 4 | 0.37s | 218.4 | 62.3s | 4/4 |
| 8 | 0.24s | 282.1 | 90.4s | 8/8 |
| 16 | 5.85s | 459.7 | 102.8s | 16/16 |
| 32 | 0.81s | 796.3 | 120.8s | 32/32 |

### 16K/2K
| 并发 | TTFT | 总吞吐 | 总延迟 | 成功率 |
|---|---|---|---|---|
| 1 | 0.11s | 71.8 | 22.2s | 1/1 |
| 2 | 0.15s | 115.5 | 14.3s | 2/2 |
| 4 | 0.16s | 204.8 | 26.0s | 4/4 |
| 8 | 0.22s | 249.3 | 28.2s | 8/8 |
| 16 | 10.72s | 339.7 | 46.3s | 16/16 |
| 32 | 0.72s | 710.2 | 47.2s | 32/32 |

### 32K/1K
| 并发 | TTFT | 总吞吐 | 总延迟 | 成功率 |
|---|---|---|---|---|
| 1 | 4.13s | 42.8 | 9.4s | 1/1 |
| 2 | 3.37s | 75.6 | 8.9s | 2/2 |
| 4 | 0.22s | 194.2 | 6.6s | 4/4 |

## 关键结论
1. 512K上下文比256K性能略优 (C=32吞吐+10%)
2. 16K输入是甜蜜点 (TTFT<0.3s, 吞吐710)
3. KV cache 完全不是瓶颈 (MLA压缩, 池子38GB只用<1%)
4. 瓶颈在算力不在显存
5. 1M上下文会OOM (MTP sampler buffer), 512K是上限

## 场景建议
| 场景 | 推荐并发 | 理由 |
|---|---|---|
| 实时对话 | C=1-4 | TTFT<1s |
| 长文档问答 | C=4-8 | TTFT<0.3s |
| 高吞吐批处理 | C=32 | 吞吐710-796 |
| Think Max推理 | C≤8 | 满足384K+上下文 |

### 32K/32K (强制输出到max_tokens, completion接口)
| 并发 | TTFT | 总吞吐 | 总延迟 avg | 入tok | 出tok | 成功率 |
|---|---|---|---|---|---|---|
| 1 | 0.13s | 97.2 | 337.1s (5.6min) | 24012 | 32768 | 1/1 |
| 2 | 3.72s | 167.9 | 364.7s (6.1min) | 48024 | 65536 | 2/2 |
| 4 | 0.26s | 271.6 | 428.9s (7.1min) | 96048 | 131072 | 4/4 |

注: 此测试用 completion + "重复文本" 任务强制输出到 max_tokens(32768),
真实对话模型通常提前 stop(写满~10K就停)。此数据反映纯 decode 算力极限。

## decode 吞吐汇总 (单请求)
| 输入 | 输出 | decode吞吐 | 说明 |
|---|---|---|---|
| 4K | 4K | 74.7 tok/s | 常规 |
| 16K | 2K | 71.8 tok/s | 长输入 |
| 32K | 32K | 97.2 tok/s | 长输出(纯decode) |

单请求 decode 稳定在 ~70-97 tok/s, 并发后总吞吐线性增长(C=32时~800 tok/s)。
