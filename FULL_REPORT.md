# DeepSeek-V4-Pro 部署与压测完整报告

> 归档于 llm-test 项目 | 2026-08 | 硬件: 2×8 RTX PRO 6000 Blackwell

## 一、测试环境

| 项目 | 配置 |
|---|---|
| 服务器 | 2× (10.10.3.15 server01 / 10.10.3.16 server02) |
| GPU | 每台 8× RTX PRO 6000 Blackwell (96GB, SM120 / CC 12.0) |
| 互联 | 8× ConnectX-8 IB (400G), 无 NVLink (PCIe P2P) |
| 以太网 | 10.10.3.x/24, 网关 10.10.3.254 |
| 软件 | vLLM 0.26.0, NCCL 2.28.9, CUDA 13.0, Ray 2.56.1 |

## 二、模型信息

| 项目 | 值 |
|---|---|
| 模型 | DeepSeek-V4-Pro (deepseek-ai/DeepSeek-V4-Pro) |
| 总参数 | 861.61B |
| 权重体积 | 864.74GB (FP4+FP8 混合 safetensors) |
| 激活参数 | 49B |
| 路由专家 | 384 个 (FP4/MXFP4), 每次激活 6 个 |
| 共享专家 | 1 个 |
| 注意力 | MLA (Multi-head Latent Attention) |
| 上下文 | max_position_embeddings = 1048576 (1M) |
| KV cache | 极小: 576 bytes/token (MLA 压缩) |

## 三、关键问题与解决 (踩坑记录)

### 3.1 SM120 乱码问题 (核心)

**现象**: 输出乱码, 如 `0.0, 0.0, 0.0...` 或无意义字符
**根因**: SM120 (PRO 6000) 上 DeepGEMM einsum 使用 SM100 的 packed scale 布局, 产生 NaN
**修复**: patch/patch_sm120.py, 改两处:
- o_proj.py: SM12x 用 SM90 风格 (1,128,128) recipe + raw f32 scales
- fp8_utils.py: SM12x 跳过 SM100 weight-scale pre-packing
**上游 PR**: https://github.com/vllm-project/vllm/pull/48052

### 3.2 MoE backend 选择

| backend | 结果 |
|---|---|
| DeepGEMM MXFP4 (auto) + patch | ✅ 正常 |
| marlin (手动指定) | ❌ 绕过 patch, 乱码 |
| 默认 (无 --moe-backend) | ✅ auto 选 DeepGEMM, patch 生效 |

注意: 后期发现某配置下 marlin 也能正常 (需配合特定参数), 但 DeepGEMM+patch 是最稳的。

### 3.3 PP (流水线并行) 乱码

TP8×PP2 配置下乱码 (即使有 patch)。patch 只修了 TP 路径。
**解决**: 用 DP2 替代 PP2, 即每台机一个 TP8 副本, 两台组成 DP2。

### 3.4 MTP 投机解码 + 长上下文 OOM

1M 上下文 + MTP 启动时, rejection sampler 的 buffer OOM:
```
sample_recovered_tokens → q = torch.empty(...) → OutOfMemoryError
```
**解决**: max-model-len 降到 512K (524288)。512K 是 MTP 模式上限。

### 3.5 Rust frontend

VLLM_USE_RUST_FRONTEND=1 时有 logprobs 解码 bug (messagepack 崩溃, 500 错误)。
最终配置仍开着 Rust frontend (性能更好), 推理正常 (乱码根因不在 frontend)。

### 3.6 Ray placement group (早期问题)

vLLM 硬编码 strategy="PACK", TP16 跨节点失败。
曾改 ray_utils.py 为 SPREAD, 后用 DP2 方案规避。

### 3.7 IP 漂移

重启后 IP 变 (DHCP)。修复:
- netplan 写静态 IP (15/16)
- 禁用 cloud-init 网络配置 (/etc/cloud/cloud.cfg.d/99-disable-network-config.cfg)
- 网卡 udev .link 规则改名 eth0 (server01 已生效, server02 待重启)

## 四、最终部署配置

### 4.1 拓扑

```
DP2 架构 (每台一个副本):
  server01 (10.10.3.15): TP8 + EP8 副本0 (8卡)
  server02 (10.10.3.16): TP8 + EP8 副本1 (8卡, headless)
  两副本通过 data-parallel 协调, 请求分流
```

### 4.2 启动参数

| 参数 | 值 | 说明 |
|---|---|---|
| tensor-parallel-size | 8 | 节点内 |
| data-parallel-size | 2 | 跨节点 |
| data-parallel-size-local | 1 | 每台1副本 |
| enable-expert-parallel | (开) | EP8, 每卡48专家 |
| moe-backend | marlin | (DeepGEMM+patch 也行) |
| kv-cache-dtype | fp8 | |
| speculative-config | mtp, num_spec=2 | 投机解码 |
| max-model-len | 524288 | 512K (1M会OOM) |
| gpu-memory-utilization | 0.95 | |
| block-size | 256 | |
| VLLM_USE_RUST_FRONTEND | 1 | |

### 4.3 启动命令

**Server01 (主)**:
```bash
source ~/vllm/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export VLLM_USE_RUST_FRONTEND=1

vllm serve deepseek-ai/DeepSeek-V4-Pro \
  --served-model-name DeepSeek-V4-Pro \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address 10.10.3.15 \
  --data-parallel-rpc-port 13345 \
  --moe-backend marlin \
  --linear-backend deep_gemm \
  --gpu-memory-utilization 0.95 \
  --max-model-len 524288 \
  --compilation-config '{}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --port 8123 \
  --api-key abc.12345
```

**Server02 (headless)**:
```bash
# 同上环境变量, vllm 命令加:
#   --headless
#   --data-parallel-start-rank 1
# 其余参数与 server01 完全一致
```



### 4.4 启动脚本 (推荐用法)

两台机已生成标准启动脚本 (含 `--max-num-batched-tokens 8192` 优化):

\`\`\`bash
# server01 (主节点)
bash /root/dsv4pro/scripts/start_server01.sh

# server02 (headless, 在 server01 启动后再跑)
bash /root/dsv4pro/scripts/start_server02.sh
\`\`\`

脚本也已归档到本仓库: `deepseek-v4-pro/scripts/`
调优参数说明见: `deepseek-v4-pro/docs/tuning_guide.md`

## 五、显存分解 (每卡 96GB)

| 项目 | 占用 | 说明 |
|---|---|---|
| 权重 | ~54 GB | FP4 专家(48个) + FP8 共享(TP8切分) |
| KV cache 池 | ~38 GB | MLA 压缩 (576 bytes/token) |
| 激活 + CUDA graph | ~5 GB | |
| 合计 | ~97 GB | 接近满载 |

**为什么 864GB 能装进 8×96GB=768GB**:
- 864GB 是所有 384 专家总和 (磁盘体积)
- EP 模式每卡只放 48 个专家 (384÷8)
- FP4 量化压缩, 每卡实际 ~54GB

## 六、压测结果

### 6.1 1K/1K (256K上下文)
| 并发 | 总吞吐(tok/s) | 总延迟 avg(s) | 成功率 |
|---|---|---|---|
| 1 | 73.8 | 13.9 | 1/1 |
| 2 | 133.0 | 15.3 | 2/2 |
| 4 | 212.6 | 18.6 | 4/4 |
| 8 | 274.0 | 29.2 | 8/8 |
| 16 | 441.8 | 35.8 | 16/16 |
| 32 | 586.4 | 54.9 | 32/32 |
| 64 | 924.2 | 69.3 | 64/64 |
| 96 | 1286.5 | 74.7 | 96/96 |
| 128 | 1462.7 | 87.1 | 128/128 |

### 6.2 2K/2K (256K上下文)
| 并发 | 总吞吐(tok/s) | 总延迟 avg(s) | 成功率 |
|---|---|---|---|
| 1 | 26.5* | 77.2* | 1/1 |
| 2 | 140.5 | 29.1 | 2/2 |
| 4 | 222.6 | 36.3 | 4/4 |
| 8 | 285.0 | 56.2 | 8/8 |
| 16 | 511.7 | 62.7 | 16/16 |
| 32 | 840.9 | 76.0 | 32/32 |

*C=1 为冷启动 (JIT 预热), 非稳态

### 6.3 4K/4K + TTFT (256K上下文)
| 并发 | TTFT(s) | 总吞吐(tok/s) | 总延迟 avg(s) | 成功率 |
|---|---|---|---|---|
| 1 | 0.50 | 71.9 | 50.2 | 1/1 |
| 2 | 0.64 | 139.7 | 54.4 | 2/2 |
| 4 | 0.13 | 213.3 | 64.8 | 4/4 |
| 8 | 0.23 | 256.0 | 85.6 | 8/8 |
| 16 | 10.04 | 434.7 | 121.1 | 16/16 |
| 32 | 11.59 | 721.8 | 140.4 | 32/32 |

### 6.4 4K/4K + TTFT (512K上下文)
| 并发 | TTFT(s) | 总吞吐(tok/s) | 总延迟 avg(s) | 成功率 |
|---|---|---|---|---|
| 1 | 0.59 | 74.7 | 47.0 | 1/1 |
| 2 | 0.69 | 114.9 | 48.8 | 2/2 |
| 4 | 0.37 | 218.4 | 62.3 | 4/4 |
| 8 | 0.24 | 282.1 | 90.4 | 8/8 |
| 16 | 5.85 | 459.7 | 102.8 | 16/16 |
| 32 | 0.81 | 796.3 | 120.8 | 32/32 |

### 6.5 16K/2K + TTFT (512K上下文)
| 并发 | TTFT(s) | 总吞吐(tok/s) | 总延迟 avg(s) | 成功率 |
|---|---|---|---|---|
| 1 | 0.11 | 71.8 | 22.2 | 1/1 |
| 2 | 0.15 | 115.5 | 14.3 | 2/2 |
| 4 | 0.16 | 204.8 | 26.0 | 4/4 |
| 8 | 0.22 | 249.3 | 28.2 | 8/8 |
| 16 | 10.72 | 339.7 | 46.3 | 16/16 |
| 32 | 0.72 | 710.2 | 47.2 | 32/32 |

### 6.6 32K/1K + TTFT (512K上下文)
| 并发 | TTFT(s) | 总吞吐(tok/s) | 总延迟 avg(s) | 成功率 |
|---|---|---|---|---|
| 1 | 4.13 | 42.8 | 9.4 | 1/1 |
| 2 | 3.37 | 75.6 | 8.9 | 2/2 |
| 4 | 0.22 | 194.2 | 6.6 | 4/4 |

### 6.7 32K/32K 强制输出 (512K上下文, completion接口)
| 并发 | TTFT(s) | 总吞吐(tok/s) | 总延迟 avg(s) | 总出tok | 成功率 |
|---|---|---|---|---|---|
| 1 | 0.13 | 97.2 | 337.1 (5.6min) | 32768 | 1/1 |
| 2 | 3.72 | 167.9 | 364.7 (6.1min) | 65536 | 2/2 |
| 4 | 0.26 | 271.6 | 428.9 (7.1min) | 131072 | 4/4 |

注: 用 completion + "重复文本" 任务强制输出到 max_tokens(32768),
真实对话模型通常提前 stop (写满~10K就停)。此数据反映纯 decode 算力极限。

## 七、横向对比分析

### 7.1 不同输入长度 (C=32, 512K上下文)
| 输入 | 输出 | TTFT | 总吞吐 | 总延迟 |
|---|---|---|---|---|
| 4K | 4K | 0.81s | 796.3 | 120.8s |
| 16K | 2K | 0.72s | 710.2 | 47.2s |
| 32K | 32K | — (C=4) | 271.6(C=4) | 428.9s(C=4) |

### 7.2 256K vs 512K 上下文 (4K/4K, C=32)
| 上下文 | 总吞吐 | 总延迟 |
|---|---|---|
| 256K | 721.8 | 140.4s |
| 512K | 796.3 | 120.8s |
| 变化 | +10% | -14% |

### 7.3 单请求 decode 吞吐
| 输入 | 输出 | decode吞吐 |
|---|---|---|
| 4K | 4K | 74.7 tok/s |
| 16K | 2K | 71.8 tok/s |
| 32K | 32K | 97.2 tok/s |

## 八、关键结论

1. **最高吞吐**: 1K/1K C=128 达 1462.7 tok/s (未现拐点)
2. **512K 优于 256K**: 同配置 C=32 吞吐 +10%, 延迟 -14%
3. **16K 是甜蜜点**: TTFT<0.3s, 吞吐710, 延迟适中
4. **32K 长输出**: 单请求 97.2 tok/s (比短输出还快, MTP 对长 decode 更有效)
5. **KV cache 无压力**: MLA 压缩至 576B/token, 池子38GB仅用<1%
6. **瓶颈在算力**: 显存满载但 KV cache 远未打满, 瓶颈是 TFLOPS
7. **1M 上下文不可行**: MTP sampler buffer OOM, 512K 是上限

## 九、场景建议

| 场景 | 推荐并发 | 输入长度 | 理由 |
|---|---|---|---|
| 实时对话 | C=1-4 | ≤4K | TTFT<1s, 延迟<65s |
| 长文档问答 | C=4-8 | 16K | TTFT<0.3s, 吞吐200-250 |
| 超长文档分析 | C=2-4 | 32K | TTFT<4s, 可接受 |
| 高吞吐批处理 | C=32-128 | 1K-4K | 吞吐800-1463 |
| Think Max 推理 | C≤8 | 16K+ | 满足384K+上下文需求 |
| 长文本生成 | C=1-4 | 32K | 97 tok/s decode, 5-7min/篇 |

## 十、文件清单

```
llm-test/
├── FULL_REPORT.md                ← 本文件 (完整报告, 自包含)
├── README.md                     ← 仓库导航
├── .gitignore
├── _template/                    ← 新测试模板
│   └── README.md
└── deepseek-v4-pro/
    ├── docs/
    │   ├── benchmark_results.md  ← 详细压测数据
    │   └── benchmark_summary.md  ← 汇总
    ├── bench_scripts/            ← 9 个压测脚本
    │   ├── bench_2k.py
    │   ├── bench_4k_ttft.py
    │   ├── bench_16k.py
    │   ├── bench_32k.py
    │   ├── bench_32k_full.py
    │   ├── bench_32k_out.py
    │   ├── bench_32k32k.py
    │   ├── bench_concurrent.py
    │   └── bench_pro.py
    ├── patch/
    │   └── patch_sm120.py        ← SM120 必需补丁
    └── logs/                     ← 25 个历史日志
        ├── test1-5_*.log         (TP16 调试)
        ├── deepseek_v4_*.log     (各种拓扑压测)
        └── glm_tp8dp2ep16_*.txt  (GPU拓扑/显存/环境快照)
```

## 十一、后续推送 GitHub

本地仓库已 git init + commit, remote 指向 https://github.com/xzwgit/llm-test.git
当前 token (fine-grained) 无创建仓库权限, 需:
1. 在 https://github.com/new 手动创建 llm-test 空仓库 (不勾README)
2. 然后: cd /root/llm-test && git push -u origin main
