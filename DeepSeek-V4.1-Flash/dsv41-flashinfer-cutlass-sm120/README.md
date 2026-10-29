# FlashInfer CUTLASS MoE Route on SM120: First Successful Deployment + DeepGEMM Comparison

> First public verification of the FlashInfer CUTLASS MoE backend (`FLASHINFER_CUTLASS_MXFP4_MXFP8`) on consumer Blackwell (SM120, RTX PRO 6000). Includes the JIT header fix, autotune timeout workaround, and same-hardware comparison against DeepGEMM.
>
> 中文版：[README.zh.md](README.zh.md)

## What was needed to make FlashInfer CUTLASS work on SM120

### 1. JIT header fix (FlashInfer wheel packaging gap)

The pip-installed FlashInfer does not bundle CUTLASS/CCCL headers needed for SM120 JIT compilation — see [flashinfer#5873](https://github.com/flashinfer-ai/flashinfer/issues/5873). Fix:

```bash
FI=$(python -c "import flashinfer, os; print(os.path.join(os.path.dirname(flashinfer.__file__),'data'))")
FLASHINFER_SRC=<path-to-flashinfer-source-with-submodules>
ln -sf $FLASHINFER_SRC/3rdparty/cutlass $FI/cutlass
ln -sf $FLASHINFER_SRC/3rdparty/cccl   $FI/cccl
ln -sf $FLASHINFER_SRC/3rdparty/spdlog $FI/spdlog
rm -rf ~/.cache/flashinfer/
```

### 2. Autotune timeout workaround

FlashInfer's first-boot autotune takes >50 minutes on SM120 (21 MXFP8 GEMM profiles), which exceeds NCCL's pair timeout — all 8 workers crash with `RuntimeError: Application timeout caused pair closure`. Filed as [flashinfer#5878](https://github.com/flashinfer-ai/flashinfer/issues/5878).

Workaround: `--no-enable-flashinfer-autotune` (same flag used in the official RTX PRO 6000 recipes for other models). Kernels still use FlashInfer's CUTLASS path — just without shape-specific tuning.

### 3. #57292 patch still required

dev511 upstream does not include the SM120 64-token page geometry. Without [#57292](https://github.com/vllm-project/vllm/pull/57292), the service fails with `block_kv == 32 or block_kv == 64` assertion. See our [dev511 upgrade page](../dsv41-flash-dev511-sm120/) for details.

## Environment

2 × 8× RTX PRO 6000 Blackwell Server Edition (SM120, cc 12.0, PCIe-only), Ubuntu 22.04.5, driver 595.71.05, CUDA 13.2.2. vLLM dev511 + FlashInfer 0.7.1 (source) + DeepGEMM 6901431 + #57292 patch.

Text + DSpark spec=5 + 128K context + TP8, single-node.

## Results: FlashInfer CUTLASS vs DeepGEMM (same hardware, same config)

| Conc | DeepGEMM tok/s | FlashInfer CUTLASS tok/s | Diff | FI TPOT | FI Acc len |
|---|---|---|---|---|---|
| c1 | 149.7 | 133.7 | -11% | 7.40ms | 2.46 |
| c4 | 343.7 | 325.7 | -5% | 11.75ms | 2.09 |
| c8 | 512.5 | **555.8** | **+8%** | 13.74ms | 2.25 |
| c16 | 744.5 | **833.1** | **+12%** | 17.98ms | 2.21 |

All tiers: zero failures. Correctness: 17×19 = 323 ✓ on both machines.

**Without autotune.** On B300 (SM100), FlashInfer with autotune is 30~54% faster than DeepGEMM. If the SM120 autotune timeout is resolved, the gap could widen further.

## Key takeaways

1. **FlashInfer CUTLASS works on SM120** — three prerequisites: JIT headers, skip autotune, #57292 patch
2. **Already better than DeepGEMM at c8+** (+8~12%) without any tuning
3. **Low concurrency slightly behind** (-5~11%) — autotune would likely close or reverse this
4. **Both routes are viable** — DeepGEMM is simpler to set up (no header fix needed), FlashInfer has more headroom if autotune works
5. **Boot time**: FlashInfer CUTLASS with cached JIT ≈ 2 min (same as DeepGEMM)

## Raw Data

```
raw/bench_c*.log       FlashInfer CUTLASS bench output (c1/c4/c8/c16)
```
