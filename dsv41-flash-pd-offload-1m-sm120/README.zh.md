# DeepSeek-V4.1-Flash 在 8×RTX PRO 6000（SM120）上：PD 分离、KV 缓存卸载到内存、1M 上下文并发

> 在**两台 8× RTX PRO 6000（SM120，消费级 Blackwell，纯 PCIe）**服务器上完成的验证配置。
> 据我们所知，本页包含 **SM120 上首个公开验证的 Prefill/Decode 分离部署**、该模型**首个视觉+PD 组合验证**，以及**首个压到实用容量极限的全 1M 上下文并发爬坡**。
>
> English version: [README.md](README.md)

## 1. 环境

| 项目 | 值 |
|---|---|
| 节点 | 2 ×（2× AMD EPYC 9A75、1.5 TB 内存、8× RTX PRO 6000 Blackwell 96 GB、CX8 400G） |
| 系统 / 驱动（测试时） | Ubuntu 22.04.5、NVIDIA 590.48.01、CUDA 13.0 |
| vLLM | 0.30.1rc1.dev382 nightly + PR #57292（SM120 64-token sparse-MLA 页几何） |
| FlashInfer / DeepGEMM | 0.7.1（main）/ main @ 6901431 |
| 模型 | DeepSeek-V4.1-Flash（476 GB FP8，视觉塔 + DSpark draft 头内置） |
| 服务形态 | 节点 A = Prefill（kv_producer）、节点 B = Decode（kv_consumer），NIXL 连接器走 400G IPoIB，前端 vllm-router |
| 基线 | NCCL 单机 8 卡 all_reduce busbw ≈ 43–44 GB/s（跨机 16 卡 48 GB/s）、P2P 矩阵全 OK |

启动要点（完整脚本见 `scripts/`，已脱敏）：

- 两角色同配置：TP8、`--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"fp8"}'`、`--kv-cache-dtype fp8`、`--engram-config '{"cpu_offload":true}'`、**双池同开** DSpark `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`（保证转移的 KV 布局兼容——已实证：双端 NIXL 兼容哈希一致）。
- KV 传输（P 侧）：`--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_ip":"<P_FABRIC_IP>","kv_port":14579,"kv_parallel_size":1}'`（D 侧 `kv_consumer`，同 `kv_ip`）。
- **Prefill 节点必须设 `VLLM_NIXL_SIDE_CHANNEL_HOST=<P_FABRIC_IP>`。** 默认是 `localhost`，会让 D 的握手回环打到自己（症状：误导性的 `Remote NIXL agent engine ID mismatch, received=<自己的 id>`）。
- Router：`vllm-router --vllm-pd-disaggregation --prefill http://<P>:8123 14579 --decode http://<D>:8123 --api-key <KEY> --prometheus-host 0.0.0.0`。**后端 P/D 不要设 `--api-key`**：router 第二段（decode）合成请求不带鉴权头，后端带 key 必 401。
- 我们测试的树早于 [vllm#57661](https://github.com/vllm-project/vllm/issues/57661)（混合注意力 KV 在 NIXL PD 下静默输出损坏）的修复。本地应用了 PR [#57662](https://github.com/vllm-project/vllm/pull/57662)（按 `(base_addr, block_len)` 去重传输区域）后，**全部测试零乱码**。

## 2. 第一部分 —— PD 分离拉起（文本 + DSpark 双池 + 视觉）

全链路（router → P 视觉编码 prefill → IB 传 KV → D DSpark decode）正确性：

- 文本：17×19 = 323 ✓
- 视觉客观图集（程序化生成：OCR / 图内算式 / 颜色计数 / 表格取数 / 双图比较 / 柱状图 / 小字 / 纯色 / 纯文本对照）：**10/10**（`raw/mm_pd_vision_log.txt`）
- D 端每个请求都有 `NIXL compatibility check passed` + `Transfer plan: local_tp=8, remote_tp=8` —— KV 真实跨机传输，不是 D 端重算。

吞吐（1K 入/1K 出、random、ignore-eos、走 router）：

| 配置 | 并发 | 出向 tok/s | TTFT 均值 | TPOT |
|---|---|---|---|---|
| PD、无投机 | 1 | 89.9 | 184 ms | 10.95 ms |
| PD、无投机 | 4 | 298.3 | 255 ms | 13.16 ms |
| **PD + DSpark 双池** | 1 | **156.8** | 147 ms | **6.24 ms** |
| **PD + DSpark 双池** | 4 | **429.1** | 208 ms | 8.91 ms |

全部档位零失败。decode 节点 DSpark 接受长度窗口 2.5–3.0。

## 3. 第二部分 —— KV 缓存卸载到 CPU 内存（SimpleCPUOffloadConnector）

官方配方选项（"Simple / CPU offload"）。单机、`kv_role: kv_both`、每 rank 25 GiB 主机池（TP8 共 200 GiB）、1M 上下文、视觉 + DSpark。

挤出-召回实验——**100 × 128K token 的互异前缀**跑两轮（同 seed；总量 13.1M tokens > GPU 池约 11M，第一轮必然挤出）：

| 轮次 | TTFT 均值 | 总时长 | 请求 |
|---|---|---|---|
| 第 1 轮（冷，全量重算） | 59,312 ms | 675.9 s | 100/100 |
| 第 2 轮（同前缀，从主机内存召回） | **255 ms** | **43.0 s** | 100/100 |

**TTFT 提升 232 倍 / 总时长提升 15.7 倍**——当逐出目的地是主机内存而非丢弃时，"100 用户 × 长上下文"场景变得可行。（原始数据：`raw/kv_offload_test/`。）

注：`LMCacheConnectorV1` 不能用于本模型（不支持混合内存架构，被连接器工厂拒绝）；`HiSparseConnector` 显式 raise "does not support DeepSeek V4"。

## 4. 第三部分 —— 1M 上下文并发

### 4.1 混合梯度（视觉+DSpark 双池 PD，max_model_len = 1,048,576，KV 池 24.08 GiB/卡）

| 档位 | 输入 tokens | 输出 tokens | 出向 tok/s | TTFT 均值 | TPOT |
|---|---|---|---|---|---|
| 1M × c1 | 2,090,060 | 2,048 | 17.3 | 48,529 ms | 10.49 ms |
| 512K × c1 | 1,048,636 | 2,048 | 24.8 | 32,263 | 8.90 |
| 512K × c2 | 2,097,272 | 4,096 | 55.9 | 22,609 | 6.92 |
| 256K × c1 | 786,522 | 3,072 | 150.0 | 291 | 6.39 |
| 256K × c2 | 1,573,044 | 6,144 | 125.6 | 6,893 | 6.66 |
| 256K × c4 | 2,097,392 | 8,192 | 166.6 | 4,798 | 11.10 |
| 128K × c1 | 524,408 | 4,096 | 169.0 | 201 | 5.73 |
| 128K × c2 | 1,048,816 | 8,192 | 234.0 | 228 | 7.65 |
| 128K × c4 | 1,573,224 | 12,288 | 240.9 | 3,979 | 9.69 |
| 128K × c8 | 2,097,632 | 16,384 | 288.0 | 3,903 | 12.55 |

每档零失败。TPOT 全档 5.7–12.6 ms，与上下文长度基本无关——decode 几乎不吃上下文代价；长上下文的成本全部落在 TTFT（prefill + 传输）。

### 4.2 全 1M 爬坡（c = 1, 2, 3, … 每档一波，直到劣化）

| c | TTFT 均值 | 出向 tok/s | 判定 |
|---|---|---|---|
| 1–13 | 37–99 s | 82–98 | 实用稳定区 |
| 14 | 217 s | 28.4 | **软塌分界**（P 端排队） |
| 16 | 315 s | 23.4 | |
| 18 | 466 s | 16.5 | |
| 20 | 605 s | 15.7 | 按约定停止 |

**c1–c20：零失败、零抢占——没有硬爆点，靠准入串行化优雅降级。** 实用上限 = c13。实测单条 1M 上下文请求 KV ≈ 1.3 GiB，池 24.08 GiB/卡。（原始数据：`raw/pd1m_full/`，含每档完整日志 + CSV + Prometheus 时序。）

### 4.3 容量公式（用于规划）

`并发 ≈ min(max_num_seqs, KV池 / (上下文长度 × ~2.2 KB/token))`——本配置：1M→约 10–13、512K→约 21、256K→约 42、128K→受 max_num_seqs 限制。对外可承诺的用户容量还受吞吐/SLO 约束（见第四部分）。

## 5. 第四部分 —— P/D 容力与配比实测

角色隔离压测（独立基准）：

- **P（prefill）极限 ≈ 52,000 tok/s**（128K 入直打 P，c32 饱和；c4 即 47K——几乎不随并发变化）
- **D（decode）极限 ≈ 1,427 tok/s**（256 入/8192 出走 router，c64，TPOT 17.3 ms）——**但 D 速率随上下文长度剧变**：1M 上下文 ~90 tok/s、128K ~290、短上下文 ~1400

平衡配比 = 单请求 P 耗时 : 单请求 D 耗时，按负载形态：

| 负载 | 配比 |
|---|---|
| 1M 入 / 1K 出（超长文档批处理） | ≈ **2P : 1D** |
| 128K 入（RAG/检索） | ≈ 1P : 1–2D |
| 短入长出（agent/对话） | ≈ 1P : 数十 D（decode 主导） |

（原始数据：`raw/pd_ratio/`。）

## 6. 工程笔记（踩过的坑）

1. `--random-input-len 1048576` + chat 模板 + 输出超出 max_model_len → HTTP 400。1M 档用 **1045000** 留余量。
2. PD router 的两段式响应**会丢投机解码统计头**——逐档 DSpark 接受率要从 decode 引擎的 `SpecDecoding metrics` 日志窗口取。
3. Prometheus：KV 用量 gauge 是 `vllm:kv_cache_usage_perc`（不是 `gpu_cache_usage_perc`）；router 指标要 `--prometheus-host 0.0.0.0`（默认绑 127.0.0.1）。
4. `vllm-router` 进程名是 `vllm::router`——`pkill -f vllm-router` 永远匹配不到。
5. P 重启后 router 健康检查有 ~60 s 窗口拒发（"Prefill policy failed to select a worker"）——重启 router 即好。

## 7. 原始数据

```
raw/
├── kv_offload_test/     round1.log、round2.log（完整官方基准输出）、卸载日志行
├── pd1m/                10 个档位摘要 + 窗口 + 2 个完整档日志 + prom/（15 条时序，10 s 步进）
├── pd1m_full/           ramp_summary.csv（c1..c20）、progress.log、tier_c13/c14/c20.log、窗口
├── pd_ratio/            8 个容量基准日志（prefill 扫描 + decode 扫描）
└── mm_pd_vision_log.txt 走 PD 全链路的视觉客观测试 10/10
scripts/                 脱敏后的启动脚本（P/D/router、梯度、卸载实验）
```

## 8. 数据边界

本页仅呈现本测试在该硬件/软件配置下的自身数据。不做也不暗示任何跨模型、跨设备、跨框架、跨量化的对比。所有指标使用 `vllm bench serve` 官方名称与单位。
