# FlashInfer CUTLASS MoE 路线在 SM120 上：首次成功部署 + DeepGEMM 对比

> 消费级 Blackwell（SM120，RTX PRO 6000）上 FlashInfer CUTLASS MoE 后端（`FLASHINFER_CUTLASS_MXFP4_MXFP8`）的首次公开验证。包含 JIT 头文件修复、autotune 超时绕过方案、以及与 DeepGEMM 的同硬件对比。
>
> English version: [README.md](README.md)

## 让 FlashInfer CUTLASS 在 SM120 上跑起来需要的三件事

### 1. JIT 头文件修复（FlashInfer wheel 打包缺口）

pip 安装的 FlashInfer 不带 SM120 JIT 编译所需的 CUTLASS/CCCL 头文件——见 [flashinfer#5873](https://github.com/flashinfer-ai/flashinfer/issues/5873)。修复：

```bash
FI=$(python -c "import flashinfer, os; print(os.path.join(os.path.dirname(flashinfer.__file__),'data'))")
FLASHINFER_SRC=<flashinfer源码路径-含子模块>
ln -sf $FLASHINFER_SRC/3rdparty/cutlass $FI/cutlass
ln -sf $FLASHINFER_SRC/3rdparty/cccl   $FI/cccl
ln -sf $FLASHINFER_SRC/3rdparty/spdlog $FI/spdlog
rm -rf ~/.cache/flashinfer/
```

### 2. Autotune 超时绕过

FlashInfer 首次启动 autotune 在 SM120 上需要 >50 分钟（21 个 MXFP8 GEMM profiles），超过 NCCL pair 超时——8 个 worker 全部崩溃 `RuntimeError: Application timeout caused pair closure`。已报 [flashinfer#5878](https://github.com/flashinfer-ai/flashinfer/issues/5878)。

绕过方法：`--no-enable-flashinfer-autotune`（与官方 RTX PRO 6000 recipes 中其他模型的做法相同）。kernel 仍然走 FlashInfer 的 CUTLASS 路径——只是没有做形状特定调优。

### 3. #57292 补丁仍然必需

dev511 上游不包含 SM120 的 64-token 页几何。没有 [#57292](https://github.com/vllm-project/vllm/pull/57292)，服务会报 `block_kv == 32 or block_kv == 64` 断言失败。详见我们的 [dev511 升级页](../dsv41-flash-dev511-sm120/)。

## 环境

2 × 8× RTX PRO 6000 Blackwell Server Edition（SM120，cc 12.0，纯 PCIe），Ubuntu 22.04.5，驱动 595.71.05，CUDA 13.2.2。vLLM dev511 + FlashInfer 0.7.1（源码编译）+ DeepGEMM 6901431 + #57292 补丁。

文本 + DSpark spec=5 + 128K 上下文 + TP8，单机。

## 结果：FlashInfer CUTLASS vs DeepGEMM（同硬件同配置）

| 并发 | DeepGEMM tok/s | FlashInfer CUTLASS tok/s | 差 | FI TPOT | FI 接受长 |
|---|---|---|---|---|---|
| c1 | 149.7 | 133.7 | -11% | 7.40ms | 2.46 |
| c4 | 343.7 | 325.7 | -5% | 11.75ms | 2.09 |
| c8 | 512.5 | **555.8** | **+8%** | 13.74ms | 2.25 |
| c16 | 744.5 | **833.1** | **+12%** | 17.98ms | 2.21 |

全部档位零失败。正确性：17×19 = 323 ✓（两台机器均验证）。

**未做 autotune。** 在 B300（SM100）上 FlashInfer 开 autotune 后比 DeepGEMM 快 30~54%。如果 SM120 的 autotune 超时问题解决，差距可能进一步拉大。

## 要点

1. **FlashInfer CUTLASS 在 SM120 上可用**——三个前提：JIT 头文件、跳过 autotune、#57292 补丁
2. **c8 以上已优于 DeepGEMM**（+8~12%）——且未做任何调优
3. **低并发略慢**（-5~11%）——autotune 有望缩小或反转
4. **两条路线都可用**——DeepGEMM 更简单（不需要修头文件），FlashInfer 有更大提升空间
5. **启动时间**：FlashInfer CUTLASS（JIT 缓存后）≈ 2 分钟（与 DeepGEMM 相同）

## 原始数据

```
raw/bench_c*.log       FlashInfer CUTLASS bench 输出（c1/c4/c8/c16）
```
