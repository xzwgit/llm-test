# FIXB：SM120 上 GLM-5.3-Flash（glm5_next）breakable CUDA graph 捕获期 IMA 修复

**状态**：本地修复，实机验证通过（2026-10-28）；上游 PR 候选。
**适用**：vLLM `0.32.1rc1.dev463`（glm5_next 混合 KDA 模型 + SM120/Blackwell + 默认 breakable cudagraph 路径）。同样的机制预计适用于相近 nightly。

## 问题

GLM-5.3-Flash（`Glm5NextForConditionalGeneration`，KDA 线性注意力 + NoPE sparse MLA 混合架构）在 SM120（RTX PRO 6000/6000D，CUDA 13 栈）上以默认配置启动时，vLLM 自动启用 breakable cudagraph（`VLLM_USE_BREAKABLE_CUDAGRAPH=1`），**CUDA graph 捕获阶段必崩**：

- 非阻塞模式：`kda.py` / `mla_attention.py` 的 `add_eager → _end_segment → capture_end → instantiate` 报 `CUDA error: an illegal memory access`（cuGraphInstantiateWithFlags 700, EVENT_RECORD node）。
- `CUDA_LAUNCH_BLOCKING=1`：稀疏 MLA eager-break 段第一个 `torch.bmm`（吸收 GEMM, `mla_attention.py:1105`）抛 `CUBLAS_STATUS_INTERNAL_ERROR`。
- **coredump 实锤真凶内核**（`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` + cuda-gdb）：`nvjet_sm120_tst_mma_64x24x64_8_16x24x64_tmaAB_alignCD4_bz_NNNN`——cuBLAS 13 在 Blackwell 上的 nvJet TMA GEMM 内核。

**机制**：breakable 图在两段捕获之间**真实执行** eager-break 里的 cuBLAS GEMM（nvJet/TMA 内核），该执行上下文触发 Device Unknown Exception；同一个 GEMM 被捕获进 FULL 图后 replay 完全正常（证明内核本身无恙、问题只在「段间 eager 执行」这个上下文）。

## 修法

`vllm/compilation/breakable_cudagraph.py` 的 `eager_break_during_capture` wrapper 加 opt-in 开关：注意力族 op（`fn.__name__` 含 `mla`）在捕获时**随段捕获**（与 FULL 模式对该 op 的既有处理一致——捕获只记录不执行），不再作为 eager-break 真实执行；KDA 的 eager-break 保留（其 recurrent 状态 op 不能进图）。

使用（补丁已应用 + venv 内 vllm 可执行）：

```bash
VLLM_BREAKABLE_CAPTURE_ATTN=1 vllm serve <GLM-5.3-Flash> --tensor-parallel-size 8 ...
```

**限制**：PD 分离（`kv_transfer` 副作用必须 eager）场景不适用——因此做成 opt-in env 而非默认行为。

## 文件

- `fixb-breakable-capture-attn.patch`：vs dev463 stock 的统一 diff（+5 行）
- `apply_fixb.py`：幂等 apply/revert/check 脚本（含 backup；路径默认 `/root/v3env`，按需改 `F=`）

## 验证（8×RTX PRO 6000 Blackwell 96G，SM120，TP8，fp8 checkpoint）

| 项 | 默认（stock） | FIXB + env |
|---|---|---|
| boot | 捕获期 IMA 崩 | **READY，PIECEWISE 106 + FULL 57 张图全捕获** |
| 探针（temperature=0） | — | 4/4 正确（算术/中文/事实/代码；终答与 eager 模式一致） |
| c1 单流（1.8K in × 1K out, ignore_eos） | — | **92.0 tok/s** |
| c32 聚合 | — | **1295.2 tok/s**（32 路延迟均匀） |

对照：`VLLM_USE_BREAKABLE_CUDAGRAPH=0` + `cudagraph_mode=FULL` 绕法可得 79.4/1179.5（FIXB 比绕法再快 16%/10%，PIECEWISE 混合批优于纯 FULL decode 图）。MTP 投机解码（`method=mtp`）在 dev463 上无论有无本补丁均于 warmup 崩（上游 #63387/#60229 未合），与本补丁无关。

## 首轮版本教训（勿重复）

1. 补丁用 `os.environ` 前确认模块级 `import os` 存在（本文件没有——用局部 `import os as _os`）。
2. env 门必须按 `fn.__name__` 含 `mla` 圈定——裸放行会连 KDA 的 eager-break 一起跳过（KDA 真不能进图，#61952 上游仍在做）。
