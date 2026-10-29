# DeepSeek-V4.1-Flash 在 vLLM dev511 Nightly（SM120）上：PD 分离、1M 上下文、完整并发套件

> 从 vLLM dev382 升级到 **dev511** 后的完整测试结果（含 mHC+AllReduce 融合 #57643、DeepSelect #56464、FlashInfer JIT 头文件修复）。
> English version: [README.md](README.md)

## 关键发现：#57292 补丁仍然必需

dev511 上游**不包含** SM120 的 64-token 页几何。没有 [#57292](https://github.com/vllm-project/vllm/pull/57292) 补丁（4 个文件：`indexer.py`、`flashinfer_sparse.py`、`sparse_mla.py`、`attention.py`），dev511 在 SM120 上会报 `block_kv == 32 or block_kv == 64` 断言失败。我们通过在第二台机器（.153）上用干净的上游代码升级验证了这一点——直到从能跑的机器（.154）复制补丁文件后才成功。

## FlashInfer JIT 修复（dev511 必需）

FlashInfer wheel 不带 SM120 JIT 编译所需的 CUTLASS/CCCL 头文件。修复方法：

```bash
FI=<site-packages>/flashinfer/data
ln -sf <flashinfer-source>/3rdparty/cutlass $FI/cutlass
ln -sf <flashinfer-source>/3rdparty/cccl   $FI/cccl
ln -sf <flashinfer-source>/3rdparty/spdlog $FI/spdlog
rm -rf ~/.cache/flashinfer/
```

## 环境

与前次测试相同的双节点集群（每节点 8× RTX PRO 6000 SM120，NIXL PD，vllm-router）。vLLM dev511 + FlashInfer 0.7.1（源码编译）+ DeepGEMM 6901431 + #57292 补丁。`--kernel-config '{"moe_backend":"deep_gemm"}'`（FlashInfer CUTLASS JIT 在 SM120 上不可靠）。

## 结果

### PD：1K 文本阶梯（random，ignore-eos，走 router）

| 并发 | tok/s | TTFT | TPOT |
|---|---|---|---|
| c1 | 178.75 | 148ms | 5.45ms |
| c4 | 439.86 | 205ms | 8.55ms |
| c8 | 597.46 | 223ms | 12.60ms |
| c16 | 840.39 | 726ms | 17.53ms |
| c32 | 1,213.74 | 418ms | 23.68ms |
| c64 | 1,766.17 | 760ms | 30.78ms |

### PD：长上下文档位（1M max_model_len）

| 档位 | tok/s | TTFT | TPOT |
|---|---|---|---|
| 128K×c1 | 87.44 | 6.4s | 5.16ms |
| 128K×c2 | 176.30 | 3.8s | 6.59ms |
| 128K×c4 | 256.44 | 4.3s | 8.03ms |
| 256K×c1 | 69.71 | 7.5s | 7.05ms |
| 256K×c2 | 152.95 | 4.3s | 7.89ms |
| 512K×c1 | 37.15 | 18.2s | 9.16ms |
| 1M×c1 | 18.72 | 48.0s | 6.57ms |

### 单机（对比用）

| 并发 | dev382 | dev511 |
|---|---|---|
| c1 | 187 | 149.7 |
| c4 | 261 | 343.7 |
| c8 | — | 512.5 |
| c16 | — | 744.5 |

### Agent（12 轮工具调用，dev511 单机）

12/12 成功，工具调用轮均 0.29s，答案轮均 0.90s，前缀缓存 73~90%，DSpark 接受率 P0=78.6%。

## 原始数据

```
raw/d511_pd_bench/     1K 阶梯档位日志 + 进度 + 窗口
raw/d511_pd_ctx/       长上下文档位日志
raw/d_engine.log       decode 节点引擎日志
raw/p_engine.log       prefill 节点引擎日志
raw/router.log         vllm-router 日志
raw/single_node/       单机测试日志
```

## 数据边界

本页仅呈现本测试在该硬件/软件配置下的自身数据。不做任何跨模型、跨设备、跨框架对比。所有指标使用 `vllm bench serve` 官方名称与单位。
