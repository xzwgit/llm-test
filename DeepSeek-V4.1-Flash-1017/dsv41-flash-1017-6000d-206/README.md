# DSV4.1-Flash-1017 on 6615X3 (8x RTX 6000D 84G, SM120) — three-leg matrix

Same-stack (vLLM 0.33.0rc1.dev243 + PR #60762 patch, FlashInfer 0.7.2.post1) three legs, all 29 tiers + c1 double-run, zero failures:

- **k7 (default sampling)** — `tiers_k7.log`
- **k7 + explicit temperature 0** (reproducible upper-bound caliber) — `tiers_k7t0.log` — headline: 4K c1 301 tok/s (acc_len 6.57), 8K c32 2192, 1K c128 2468
- **no speculative** (capture 512, KV pool 15.34M tokens) — `tiers_nospec.log` — flat c1 69-73 tok/s across lengths

KV pools: k7 legs 11,606,172 tokens (11.07x@1M); no-spec 15,344,466 (14.63x). Host RAM peak (10s sampling, full run): used 341 GiB / vLLM process RSS 335 GiB (cpu_offload expert residency; no-spec leg RSS ≈306 GiB ≈ draft weights delta).

BOOM CMD summaries in `bootcmd_*.txt`; BOOT CMD in log headers. Raw timeline + memory csv archived locally.
