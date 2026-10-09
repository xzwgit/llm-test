# Rescue archive 2026-10-09 — previously-unarchived campaign data

Raw logs/results rescued from fleet boxes before further /root cleanups. None of this
was previously pushed here; distilled numbers exist only in issue/PR comments and
internal notes.

## rescue-156.tar.gz (10.10.3.156, server06)
- `dspark_ab/` — DSpark x independent-autotune A/B (2026-10-07, PR flashinfer#6161 validation):
  baseline arm bench logs (c1/c2-c32) + remnants of the DSpark arm
- `base_bench/`, `moe_ab_cutlass/` — default-MoE vs cutlass MoE A/B per-tier logs (dev738, 2026-10-06/07)
- `cake_bench/`, `cake2_bench/` — FlashInfer cake kernel E2E A/B logs, incl. the 10-05 fused-writer
  retest (data referenced by the #5690 draft comment that was never posted)
- `h3_logs/`, `h3_out/*.txt` — MiniMax H3 real-time video reproduction logs + response headers
  (mp4 outputs excluded); `mimo-v26-flash/*` top-level deploy/test logs (vLLM-Omni / MiMo campaign)

## rescue-154.tar.gz (10.10.3.154, server04)
- `pd1m/`, `pd1m_full/`, `pd_ratio/` — P/D disaggregation 1M-context tiers, c1-c20 ramp,
  P/D capacity-ratio measurements (2026-10-01)
- `kv_offload_test/` — SimpleCPUOffloadConnector evict/recall rounds (232x TTFT)
- `nccl_baseline_154/` — single-node NCCL baseline (busbw 43-45 GB/s)
- `mm_test_out/` — vision objective-suite outputs
- `vllm31/cell_*.log`, `matrix_run.log` — 2026-10-09 SM120 block-size matrix
  (stock fail + uniform-64/per-layer x DeepGEMM pr14/1e1842a, all 323) — raw evidence for
  vllm#60762 and the vllm#59203 matrix comment
- `vllm31/flashnext_fp8_boot.log` — Qwen3.8-Flash-Next-FP8 TP8+EP+MTP first boot snapshot
  (2026-10-09, MTP acceptance 75%/len 3.25, autotune non-blocking via #6161 patch)

## rescue-153.tar.gz (10.10.3.153, server03)
- `mm_test_out/` — vision suite outputs from the .153 VL arm

Note: .92's qwen27b-bench raw dirs were already cleaned before this rescue; the GB10
campaign reports in this repo carry the distilled data.
