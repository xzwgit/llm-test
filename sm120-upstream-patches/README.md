# SM120 Upstream Patches

> Verified patches and evidence we submitted upstream after running DeepSeek-V4.1-Flash serving on 8×RTX PRO 6000 (SM120, consumer Blackwell, 100 KB shared memory per block).
> Each subdirectory holds the patch as submitted plus the verification evidence from our machines.
>
> 中文版：[README.zh.md](README.zh.md)

| Patch | Upstream | Type | Status at time of archiving |
|---|---|---|---|
| NIXL PD handshake loopback diagnostic | [vllm-project/vllm#59583](https://github.com/vllm-project/vllm/pull/59583) | code | open |
| NIXL P/D troubleshooting docs (auth layering, side-channel loopback) | [vllm-project/vllm#59584](https://github.com/vllm-project/vllm/pull/59584) | docs | open |
| `vllm bench serve --metrics-url` (spec-decode stats behind a PD router) | [vllm-project/vllm#59587](https://github.com/vllm-project/vllm/pull/59587) | code | open |
| Engram sinkhorn `step_reduce` fits 100 KB SMEM + test tolerance fix | [deepseek-ai/TileKernels#37](https://github.com/deepseek-ai/TileKernels/pull/37) | code+test | open (note: TileKernels has merged 0 external PRs since inception — we also carry this as a vendored patch) |

Related upstream artifacts we filed with evidence (no patch): [TileKernels#36](https://github.com/deepseek-ai/TileKernels/issues/36) (sinkhorn/SM120 failure inventory), verification comment on [FlashMLA#124](https://github.com/deepseek-ai/FlashMLA/issues/124) (SM120 compiles, blocked by 227 KB SMEM plan).

## Directory layout

```
vllm-59583-nixl-loopback-diagnostic/      patch.diff + verification.txt
vllm-59584-nixl-pd-troubleshooting-docs/  patch.diff
vllm-59587-bench-metrics-url/             patch.diff + verification.txt
tilekernels-37-sinkhorn-smem/             kernel.patch + test.patch + verification.txt + tolerance_simulation.py
```

## Why these patches exist (the common thread)

The recurring wall on SM120 is the **100 KB shared-memory budget per block** (228 KB on datacenter sm_100/sm_103) — kernels and infrastructure sized for datacenter parts fail at launch or silently misbehave. Beyond that: PD-disaggregation operational gaps that only appear when you actually deploy two nodes. All four patches were verified end-to-end on our 2-node SM120 cluster before submission; none is speculative.

The TileKernels patch keeps the original accumulation order (bitwise-identical results on datacenter parts) and only reduces the copy-pipeline depth when the device budget requires it; the accompanying test fix replaces a default-tolerance comparison that a pure-torch row-serial loop already exceeds at num_partials=564 (a shape only consumer Blackwell's 188-SM count generates).

## vllm PR #57662 — NIXL region dedup (mixed-attention silent corruption, fixes #57661)

Upstream PR by @G-haoyu; we applied it to our 2-node SM120 P/D setup (vLLM dev382) and verified
zero corruption with the full-feature config (DSpark in both pools + vision + KV fp8):
16+ benchmark tiers, vision objective suite 10/10, long-form Chinese clean.

- `pr57662.diff` — the patch as applied to `base_worker.py` (on-box backups: `.bak57661/.orig`)
- `patch_57661.py` — the applier script we used

Note: the PR entered merge-conflict state on 2026-09-28 and awaits the author's rebase;
this diff is the bridge we run locally until it lands.

## glm53-flash-fixb-breakable-capture-attn

本地修复（上游 PR 候选）：SM120 上 GLM-5.3-Flash（glm5_next）默认 breakable cudagraph 捕获期 IMA——eager-break 段间真实执行的 cuBLAS nvJet TMA GEMM 崩（coredump 命名）。修法=opt-in env `VLLM_BREAKABLE_CAPTURE_ATTN=1` 让注意力族 op 随段捕获（KDA break 保留）。验证：PIECEWISE 106+FULL 57 图全捕获，探针 4/4，c1 92.0 / c32 1295.2 tok/s（8×PRO 6000 96G TP8）。详见子目录 README。
