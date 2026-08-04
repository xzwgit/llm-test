# DeepSeek-V4-Pro 压测完整报告

> 配置: DP2(双副本TP8) + MTP(num_spec=2) + marlin, 2×8 RTX PRO 6000 (96GB, SM120), 512K上下文
> 工具: bench_ultimate.py (completion 强制输出 + vLLM metrics 差值)

## 指标说明
- **TTFT**: 首 token 延迟
- **ITL**: token 间延迟 (Inter-Token Latency, ms/token)
- **Output tput**: decode 生成速度 (核心性能指标)
- **Total tput**: 输入+输出综合吞吐
- **MTP%**: MTP 投机解码接受率
- **PC%**: prefix cache 命中率

---

## 一、终极版压测 (bench_ultimate.py, 含 ITL/MTP/PC)

### 1.1 4K/4K (输入684tok, 输出4096tok)
| C | TTFT(s) | Latency(s) | ITL(ms) | Output tput | Total tput | MTP% | PC% |
|---|---|---|---|---|---|---|---|
| 1 | 0.23 | 44.0 | 10.7 | 93.1 | 108.6 | 84.5 | 0.0 |
| 4 | 0.43 | 52.5 | 12.7 | 308.8 | 360.4 | 86.1 | 56.1 |
| 16 | 0.61 | 81.2 | 19.7 | 792.7 | 925.1 | 88.1 | 74.9 |
| **32** | 7.11 | 101.9 | 23.1 | **1263.5** | **1474.6** | 88.6 | 74.9 |

### 1.2 16K/16K (输入7012tok, 输出16384tok)
| C | TTFT(s) | Latency(s) | ITL(ms) | Output tput | Total tput | MTP% | PC% |
|---|---|---|---|---|---|---|---|
| 1 | 0.11 | 178.9 | 10.9 | 91.6 | 130.8 | 83.9 | 98.6 |
| 2 | 0.12 | 177.4 | 10.8 | 184.7 | 263.8 | 84.4 | 98.6 |
| 4 | 0.13 | 213.8 | 13.0 | 303.7 | 433.7 | 83.6 | 98.6 |
| 8 | 0.27 | 318.0 | 19.4 | 407.9 | 582.4 | 83.5 | 98.6 |
| **16** | 0.34 | 339.3 | 20.7 | **765.8** | **1093.5** | 83.2 | 98.6 |

### 1.3 32K/32K (输入14292tok, 输出32768tok)
| C | TTFT(s) | Latency(s) | ITL(ms) | Output tput | Total tput | MTP% | PC% |
|---|---|---|---|---|---|---|---|
| 1 | 2.21 | 361.1 | 11.0 | 90.8 | 130.3 | 84.0 | 48.4 |
| 2 | 1.23 | 358.8 | 10.9 | 182.0 | 261.4 | 84.3 | 73.4 |
| 4 | 0.22 | 429.0 | 13.1 | 302.6 | 434.6 | 83.3 | 98.5 |

---

## 二、补充场景 (早期测试, chat 接口, 含 1K/2K/短输出)

### 2.1 1K/1K (256K上下文)
| C | 总吞吐(tok/s) | 总延迟(s) | 成功率 |
|---|---|---|---|
| 1 | 73.8 | 13.9 | 1/1 |
| 4 | 212.6 | 18.6 | 4/4 |
| 32 | 586.4 | 54.9 | 32/32 |
| 64 | 924.2 | 69.3 | 64/64 |
| 128 | 1462.7 | 87.1 | 128/128 |

### 2.2 2K/2K (256K上下文)
| C | 总吞吐(tok/s) | 总延迟(s) | 成功率 |
|---|---|---|---|
| 2 | 140.5 | 29.1 | 2/2 |
| 8 | 285.0 | 56.2 | 8/8 |
| 32 | 840.9 | 76.0 | 32/32 |

### 2.3 256K vs 512K 上下文对比 (4K/4K, C=32)
| 上下文 | 总吞吐 | 总延迟 |
|---|---|---|
| 256K | 721.8 | 140.4s |
| 512K | 796.3 | 120.8s |
512K 比 256K 吞吐 +10%, 延迟 -14%。

---

## 三、关键结论

### 1. 单请求 decode 极稳定 ~91 tok/s
无论输入长度 (4K/16K/32K), 单请求 Output tput 稳定 90-93 tok/s, ITL ~11ms/token。

### 2. MTP 接受率 83-89%
num_speculative_tokens=2, 接受率 83-89% → 平均每步产出 ~1.8 token。
这是 decode ~91 tok/s 远超无 MTP (~50 tok/s) 的原因。并发越高接受率略升。

### 3. Prefix Cache 效果显著
- 重复 prompt: 98.6% 命中
- 首次请求: 0% (冷启动)
- Agent 多轮场景受益巨大 (system prompt 复用)

### 4. 最高吞吐
- 4K/4K C=32: **Total 1474.6 tok/s** (历史最高)
- 1K/1K C=128: 1462.7 tok/s
- 16K/16K C=16: 1093.5 tok/s

### 5. 瓶颈在算力不在显存
显存满载 (~97GB/卡) 但 KV cache 池 (~38GB) 用量 <1% (MLA 压缩 576B/token)。

---

## 四、场景建议

| 场景 | 推荐并发 | 输入 | 理由 |
|---|---|---|---|
| 实时对话 | C=1-4 | ≤4K | TTFT<1s |
| AI Agent 多轮 | C=4-8 | 4-16K | PC命中, TTFT低 |
| 长文档问答 | C=4-8 | 16K | TTFT<0.3s |
| 超长文档分析 | C=2-4 | 32K | TTFT<4s |
| 高吞吐批处理 | C=32+ | 1K-4K | 吞吐1400+ |
| 长文本生成 | C=1-4 | 32K | 91 tok/s, 6min/32K篇 |

---

## 五、复测命令
```bash
# 4K/4K
python bench_ultimate.py --in-len 12 --out-tokens 4096 -c 1 4 16 32
# 16K/16K
python bench_ultimate.py --in-len 125 --out-tokens 16384 -c 1 2 4 8 16
# 32K/32K
python bench_ultimate.py --in-len 255 --out-tokens 32768 -c 1 2 4
```

---

## 六、官方工具 vllm bench serve 数据 (4K/4K)

> 工具: `vllm bench serve` (官方) + `vllm_bench_wrap.sh` (封装)
> 数据集: random + `--ignore-eos` (强制输出, 随机token, 标准测法)
> 与上面 bench_ultimate.py 的差异见第七节

### 6.1 4K/4K 完整数据 (含 TPOT/p99)

| C | Output tput | Total tput | TTFT mean(ms) | TTFT p99(ms) | TPOT mean(ms) | ITL mean(ms) | MTP% |
|---|---|---|---|---|---|---|---|
| 1 | 69.9 | 139.9 | 159 | 184 | 14.3 | 29.3 | 52.5 |
| 4 | 173.3 | 346.7 | 746 | 1341 | 19.3 | 35.9 | 42.3 |
| 16 | 411.5 | 823.5 | 2892 | 5558 | 34.2 | 64.2 | 43.5 |
| 32 | 407.0 | 814.5 | 19485 | 22826 | 53.0 | 96.1 | 37.5 |

新增指标说明:
- **TPOT** (Time Per Output Token): 每个输出token的平均耗时(不含首token), 越低越好
- **TTFT p99**: 99分位首token延迟 (尾部延迟, 反映最慢请求)
- **Peak concurrent**: 实际峰值并发 (受调度影响可能略高于设定值)

### 6.2 关键观察 (官方工具)

1. **C=16→32 吞吐不再增长** (823→814), 说明 4K/4K 在 C=16 已达算力饱和
2. **TPOT 随并发恶化**: C=1 14.3ms → C=32 53ms (3.7倍), 因 decode 排队
3. **TTFT p99 尾部延迟严重**: C=32 达 22.8s (部分请求等很久才轮到 prefill)
4. **MTP 接受率 37-52%**: 随机token难预测, position0接受60%, position1仅14-30%

---

## 七、两套工具差异分析 (重要)

### 7.1 数据对比 (4K/4K)

| 指标 | bench_ultimate.py | vllm bench serve | 差异原因 |
|---|---|---|---|
| **MTP 接受率** | 84-89% | 37-52% | ★ 核心差异 (见下) |
| **C=1 Output tput** | 93.1 tok/s | 69.9 tok/s | MTP接受多→decode快 |
| **C=32 Output tput** | 1263.5 tok/s | 407.0 tok/s | 同上, 差距放大 |
| **Prefix cache** | 0-75% | 0% | ① |
| **TPOT** | 未测 | 14-53ms | ② |

### 7.2 差异根因

**① MTP 接受率差异 (核心)**

| 工具 | 输出内容 | 草稿模型可预测性 | 接受率 |
|---|---|---|---|
| bench_ultimate.py | "重复文本" (语义连贯) | 高 (模式重复, 易猜) | 84-89% |
| vllm bench serve | `--ignore-eos` 随机token | 低 (无语义, 难猜) | 37-52% |
| **真实对话** | 正常语言 | 中等 | **约 60-70%** (估) |

MTP 投机解码的接受率高度依赖"草稿模型能否猜对下一个token":
- 重复文本: 草稿模型学到模式, 几乎全猜对 → 接受率高 → decode 快
- 随机token: 草稿模型猜不对 → 接受率低 → 退化为单步decode → 慢
- 真实场景介于两者之间

**② Prefix cache 差异**
- bench_ultimate.py: 用固定prompt重复测试 → prefix cache 命中率高 (相同前缀复用)
- vllm bench serve: random dataset 每次prompt不同 → prefix cache 0% (标准公平测法)

**③ 数据集差异**
- bench_ultimate.py: 真实prompt内容 (虽是重复段落, 但有语义)
- vllm bench serve: 随机token填充 (无语义, 纯测算力)

### 7.3 哪个数据更可信

**取决于测试目的**:
| 目的 | 用哪个工具 | 理由 |
|---|---|---|
| **算力上限测试** (公平对比硬件) | vllm bench serve | 标准化, 无语义加成, 可复现 |
| **真实场景预估** | bench_ultimate.py | 含prefix cache和MTP真实效果 |
| **横向对比其他模型** | vllm bench serve | 官方工具, 业界通用 |
| **Agent多轮场景** | bench_ultimate.py | prefix cache 反映真实agent行为 |

### 7.4 结论

两套工具数据**都有效, 反映不同侧面**:
- vllm bench 的 4K/4K C=16 总吞吐 823 tok/s 是**纯算力基线** (无MTP/PC加成)
- bench_ultimate 的 4K/4K C=16 总吞吐 925 tok/s 含**真实场景加成** (MTP 88% + PC 75%)
- 真实部署性能介于两者之间, 偏向 bench_ultimate (因正常对话MTP接受率约60-70%)
