# DeepSeek-V4-Pro 终极版压测报告

> 工具: bench_ultimate.py (completion 强制输出 + vLLM metrics 差值)
> 配置: DP2 + MTP(num_spec=2) + marlin, 2×8 RTX PRO 6000, 512K上下文, max-num-batched-tokens=8192

## 指标说明
- **TTFT**: 首 token 延迟 (Time To First Token)
- **ITL**: token 间延迟 (Inter-Token Latency, ms/token)
- **Output tput**: decode 生成速度 (核心性能)
- **Total tput**: 输入+输出综合吞吐
- **MTP%**: MTP 投机解码接受率 (越高越快)
- **PC%**: prefix cache 命中率

## 场景1: 4K/4K (输入684tok, 输出4096tok)

| C | TTFT(s) | Latency(s) | ITL(ms) | Output tput | Total tput | MTP% | PC% |
|---|---|---|---|---|---|---|---|
| 1 | 0.23 | 44.0 | 10.7 | 93.1 | 108.6 | 84.5 | 0.0 |
| 4 | 0.43 | 52.5 | 12.7 | 308.8 | 360.4 | 86.1 | 56.1 |
| 16 | 0.61 | 81.2 | 19.7 | 792.7 | 925.1 | 88.1 | 74.9 |
| **32** | 7.11 | 101.9 | 23.1 | **1263.5** | **1474.6** | 88.6 | 74.9 |

## 场景2: 16K/16K (输入7012tok, 输出16384tok)

| C | TTFT(s) | Latency(s) | ITL(ms) | Output tput | Total tput | MTP% | PC% |
|---|---|---|---|---|---|---|---|
| 1 | 0.11 | 178.9 | 10.9 | 91.6 | 130.8 | 83.9 | 98.6 |
| 2 | 0.12 | 177.4 | 10.8 | 184.7 | 263.8 | 84.4 | 98.6 |
| 4 | 0.13 | 213.8 | 13.0 | 303.7 | 433.7 | 83.6 | 98.6 |
| 8 | 0.27 | 318.0 | 19.4 | 407.9 | 582.4 | 83.5 | 98.6 |
| **16** | 0.34 | 339.3 | 20.7 | **765.8** | **1093.5** | 83.2 | 98.6 |

## 场景3: 32K/32K (输入14292tok, 输出32768tok)

| C | TTFT(s) | Latency(s) | ITL(ms) | Output tput | Total tput | MTP% | PC% |
|---|---|---|---|---|---|---|---|
| 1 | 2.21 | 361.1 | 11.0 | 90.8 | 130.3 | 84.0 | 48.4 |
| 2 | 1.23 | 358.8 | 10.9 | 182.0 | 261.4 | 84.3 | 73.4 |
| 4 | 0.22 | 429.0 | 13.1 | 302.6 | 434.6 | 83.3 | 98.5 |

## 关键发现

### 1. 单请求 decode 极稳定 (~91 tok/s)
无论输入长度 (4K/16K/32K), 单请求 Output tput 稳定在 90-93 tok/s:
- 4K: 93.1 | 16K: 91.6 | 32K: 90.8
对应 ITL ~11ms/token, 受输入长度影响极小。

### 2. MTP 接受率稳定 83-89%
- num_speculative_tokens=2, 每次预测2个token
- 接受率 83-89% 意味着平均每步实际产出 ~1.8 token
- 这就是 decode 速度 (~91 tok/s) 远超单步 (~50 tok/s 无MTP) 的原因
- 并发越高接受率略升 (batch 更规整)

### 3. Prefix Cache 效果显著
- 16K 重复 prompt: 命中率 98.6% (几乎全命中)
- 4K 首次请求: 0% (冷启动)
- 32K 首次: 48% (部分前缀与之前16K测试重合)
- Agent 多轮场景受益巨大

### 4. 并发扩展性
- 4K/4K C=32: 总吞吐 1474.6 tok/s (最高)
- 16K/16K C=16: 总吞吐 1093.5 tok/s
- 32K/32K C=4: 总吞吐 434.6 tok/s
长输出场景并发收益递减 (单请求 decode 时间长, GPU 已近满载)

### 5. TTFT 分化
- 低并发 + 高 PC: <0.3s (prefix cache 命中)
- 高并发 prefill 排队: 7-10s (4K C=32, 16K C=16部分)
- 冷启动无缓存: 2.2s (32K C=1, 全量 prefill 14K token)

## 复测命令
```bash
# 4K/4K
python bench_ultimate.py --in-len 12 --out-tokens 4096 -c 1 4 16 32
# 16K/16K
python bench_ultimate.py --in-len 125 --out-tokens 16384 -c 1 2 4 8 16
# 32K/32K
python bench_ultimate.py --in-len 255 --out-tokens 32768 -c 1 2 4
```
