# DeepSeek-V4.1-Flash on 8×RTX PRO 6000 (SM120): PD Disaggregation, KV CPU Offload, and 1M-Context Concurrency

> Verified configurations on **two single-socket-dual nodes of 8× RTX PRO 6000 (SM120, consumer Blackwell, PCIe-only)**.
> To our knowledge this includes the **first public verification of Prefill/Decode disaggregation on SM120**, the **first vision+PD combination for this model**, and the **first full-1M-context concurrency ramp to the practical capacity limit**.
>
> 中文版：[README.zh.md](README.zh.md)

## 1. Environment

| Item | Value |
|---|---|
| Nodes | 2 × (2× AMD EPYC 9A75, 1.5 TB RAM, 8× RTX PRO 6000 Blackwell 96 GB, CX8 400G) |
| OS / driver (at test time) | Ubuntu 22.04.5, NVIDIA 590.48.01, CUDA 13.0 |
| vLLM | 0.30.1rc1.dev382 nightly + PR #57292 (SM120 64-token sparse-MLA page geometry) |
| FlashInfer / DeepGEMM | 0.7.1 (main) / main @ 6901431 |
| Model | DeepSeek-V4.1-Flash (476 GB FP8, vision tower + DSpark draft head in checkpoint) |
| Serving layout | Node A = Prefill (kv_producer), Node B = Decode (kv_consumer), NIXL connector over 400G IPoIB, vllm-router front-end |
| Baseline sanity | NCCL 8-GPU all_reduce busbw ≈ 43–44 GB/s per node (16-GPU cross-node 48 GB/s), P2P matrix all OK |

Launch essentials (full scripts in `scripts/`, sanitized):

- Both roles: TP8, `--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"fp8"}'`, `--kv-cache-dtype fp8`, `--engram-config '{"cpu_offload":true}'`, DSpark `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'` **in both pools** (keeps transferred KV layout compatible — verified: identical NIXL compatibility hash on both ends).
- KV transfer (P side): `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_ip":"<P_FABRIC_IP>","kv_port":14579,"kv_parallel_size":1}'` (D side: `kv_consumer`, same `kv_ip`).
- **`VLLM_NIXL_SIDE_CHANNEL_HOST=<P_FABRIC_IP>` must be set on the Prefill node.** Default is `localhost`, which makes D's handshake loop back to itself (symptom: misleading `Remote NIXL agent engine ID mismatch, received=<own id>`).
- Router: `vllm-router --vllm-pd-disaggregation --prefill http://<P>:8123 14579 --decode http://<D>:8123 --api-key <KEY> --prometheus-host 0.0.0.0`. **Backends must NOT set `--api-key`**: the router's second-stage (decode) request does not carry the auth header and would get 401.
- The tree predates the fix for [vllm#57661](https://github.com/vllm-project/vllm/issues/57661) (silent output corruption for hybrid-attention KV with NIXL PD). We applied PR [#57662](https://github.com/vllm-project/vllm/pull/57662) (dedup transfer regions by `(base_addr, block_len)`) locally — with it, **zero garbling across all tests**.

## 2. Part 1 — PD disaggregation bring-up (text + DSpark dual-pool + vision)

Correctness through the full chain (router → P vision-encoder prefill → KV transfer over IB → D DSpark decode):

- Text: 17×19 = 323 ✓
- Vision objective suite (program-generated images: OCR / in-image math / color counting / table cell / two-image comparison / bar chart / small text / solid color / text control): **10/10** (`raw/mm_pd_vision_log.txt`)
- D-side workers report `NIXL compatibility check passed` + `Transfer plan: local_tp=8, remote_tp=8` on every request — KV is genuinely transferred, not recomputed on D.

Throughput (1K-in/1K-out, random, ignore-eos, via router):

| Config | Conc | Output tok/s | TTFT mean | TPOT |
|---|---|---|---|---|
| PD, no spec decode | 1 | 89.9 | 184 ms | 10.95 ms |
| PD, no spec decode | 4 | 298.3 | 255 ms | 13.16 ms |
| **PD + DSpark dual-pool** | 1 | **156.8** | 147 ms | **6.24 ms** |
| **PD + DSpark dual-pool** | 4 | **429.1** | 208 ms | 8.91 ms |

All tiers: 0 failed requests. DSpark acceptance windows on the decode node: 2.5–3.0.

## 3. Part 2 — KV cache offload to CPU memory (SimpleCPUOffloadConnector)

Official recipe option ("Simple / CPU offload"). Single node, `kv_role: kv_both`, 25 GiB host pool per rank (200 GiB total across TP8), 1M max len, vision + DSpark.

Evict-recall experiment — two identical rounds of **100 × 128K-token distinct prompts** (same seed; 13.1M total tokens > the ~11M-token GPU pool, so round 1 must evict):

| Round | Mean TTFT | Wall time | Requests |
|---|---|---|---|
| 1 (cold, full recompute) | 59,312 ms | 675.9 s | 100/100 |
| 2 (same prefixes, recalled from host DRAM) | **255 ms** | **43.0 s** | 100/100 |

**232× TTFT improvement / 15.7× wall-clock improvement** — the "100 users × long context" scenario becomes viable when eviction destination is host memory instead of drop. (Raw: `raw/kv_offload_test/`.)

Note: `LMCacheConnectorV1` cannot be used with this model (no hybrid-memory-architecture support → rejected by the connector factory); `HiSparseConnector` explicitly raises "does not support DeepSeek V4".

## 4. Part 3 — 1M-context concurrency

### 4.1 Mixed ladder (vision+DSpark dual-pool PD, max_model_len = 1,048,576, KV pool 24.08 GiB/card)

| Tier | Input tokens | Output tokens | Output tok/s | TTFT mean | TPOT |
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

Zero failures on every tier. TPOT stays 5.7–12.6 ms regardless of context length — decode is nearly insensitive to context; all long-context cost lands on TTFT (prefill + transfer).

### 4.2 Full-1M ramp (c = 1, 2, 3, … one wave per tier, until degradation)

| c | TTFT mean | Output tok/s | Verdict |
|---|---|---|---|
| 1–13 | 37–99 s | 82–98 | practical stable region |
| 14 | 217 s | 28.4 | **soft-collapse boundary** (P-side queueing) |
| 16 | 315 s | 23.4 | |
| 18 | 466 s | 16.5 | |
| 20 | 605 s | 15.7 | stopped by design |

**c1–c20: zero failed requests, zero preemptions — no hard failure; graceful degradation via admission serialization.** Practical limit = c13. Measured KV per 1M-context request ≈ 1.3 GiB (empirical), pool 24.08 GiB/card. (Raw: `raw/pd1m_full/` incl. per-tier full logs + CSV + Prometheus series.)

### 4.3 Capacity formula (for sizing)

`concurrency ≈ min(max_num_seqs, KV_pool / (context_len × ~2.2 KB/token))` — with this pool: 1M→~10–13, 512K→~21, 256K→~42, 128K→max_num_seqs-bound. The user-facing capacity is throughput/SLO-bound, not just KV-bound (see Part 4).

## 5. Part 4 — P/D capacity and ratio measurement

Isolated role saturation (separate benches):

- **P (prefill) ceiling ≈ 52,000 tok/s** (128K-in direct-to-P, saturates at c32; c4 already 47K — nearly concurrency-independent)
- **D (decode) ceiling ≈ 1,427 tok/s** (256-in/8192-out via router, c64, TPOT 17.3 ms) — **but D's rate is context-length dependent**: ~90 tok/s at 1M ctx, ~290 at 128K, ~1400 short ctx

Balanced ratio = per-request P time : per-request D time, hence by workload shape:

| Workload | Ratio |
|---|---|
| 1M-in / 1K-out (long-document batch) | ≈ **2P : 1D** |
| 128K-in (RAG/retrieval) | ≈ 1P : 1–2D |
| short-in long-out (agent/chat) | ≈ 1P : tens of D (decode-dominated) |

(Raw: `raw/pd_ratio/`.)

## 6. Engineering notes (gotchas found the hard way)

1. `--random-input-len 1048576` + chat template + output exceeds max_model_len → HTTP 400. Use **1045000** for the 1M tier.
2. The PD router's two-stage response **drops speculative-decoding stat headers** — per-tier DSpark acceptance must be read from the decode engine's `SpecDecoding metrics` log windows.
3. Prometheus: the KV-usage gauge is `vllm:kv_cache_usage_perc` (not `gpu_cache_usage_perc`); router metrics need `--prometheus-host 0.0.0.0` (default binds 127.0.0.1).
4. `vllm-router` process name is `vllm::router` — `pkill -f vllm-router` never matches it.
5. After a P restart, the router health-check has a ~60 s window that rejects requests ("Prefill policy failed to select a worker") — restart the router.

## 7. Raw data

```
raw/
├── kv_offload_test/     round1.log, round2.log (full official bench output), offload lines
├── pd1m/                10 tier summaries + window + 2 full tier logs + prom/ (15 series, 10 s step)
├── pd1m_full/           ramp_summary.csv (c1..c20), progress.log, tier_c13/c14/c20.log, window
├── pd_ratio/            8 capacity-bench logs (prefill sweep + decode sweep)
└── mm_pd_vision_log.txt vision-objective 10/10 through the PD chain
scripts/                 sanitized launch scripts (P/D/router, ladders, offload test)
```

## 8. Data boundary

This page presents only this test's own data on this hardware/software configuration. No cross-model, cross-device, cross-framework, or cross-quantization comparisons are made or implied. All metrics use official `vllm bench serve` names and units.
