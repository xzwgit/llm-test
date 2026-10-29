# GLM-5.3-Flash on RTX PRO 6000D (84G, SM120) x8 TP8

Same two-leg recipe as the PRO 6000 (96G) campaign (`../glm53-flash-pro6000-sm120/`),
cross-SKU comparison: 84G cards give 27.98 GiB/card KV cache vs 38.71 GiB on 96G
(weights 41.31 GiB/card on both).

## 2026-10-28 · two-leg 29-tier matrix (`logs/matrix-157/`)

| Path | Content |
|---|---|
| `logs/matrix-157/matrix_k5.log` | Raw bench output, MTP k5 leg, 29 tiers + c1 double-run |
| `logs/matrix-157/matrix_nospec.log` | Raw bench output, no-spec leg, 29 tiers + c1 double-run |
| `logs/matrix-157/spec_k5.log` | Engine-side SpecDecoding metrics |
| `logs/matrix-157/master.log` | Driver trace incl. BOOT CMD lines |
| `logs/matrix-157/deploy.log`, `deploy_and_run.sh` | Chained deploy (env + weights rsync) then matrix |
| `logs/matrix-157/boot_k{5}.log`, `boot_nospec.log` | Server boot logs (host addr sanitized) |
| `logs/matrix-157/glm_matrix.sh` | Driver script |
| `logs/matrix-157/tiers.tsv` | Parsed per-tier throughput/latency table |
| `logs/matrix-157/acceptance_k5.tsv` | Parsed per-tier acceptance (bench window sum) |

Key numbers (output tok/s, k5 vs nospec, c1 = second run):
- Single stream c1: 1K 78.0/78.3 (1.00x), 4K 202.2/83.2 (2.43x), 8K 149.2/85.0 (1.75x), 32K 327.0/84.6 (3.86x), 64K 267.0/84.5 (3.16x)
- High concurrency: 1K c128 nospec wins (1764.9 vs 1265.9); 64K c8 k5 1.88x (756.6 vs 402.4)
- No-spec single-stream identical to 96G SKU (~78-85 flat); k5 single-stream lower than 96G at mid bands, driven by acceptance variance (8K c1 41% here vs 94.6% on 96G; run-to-run variance present, server-side default sampling is stochastic)
- Acceptance peaks at 32K band (98.5%/5.92 c1); zero failures both legs

## 2026-10-29 · two-leg matrix on GB202 platform (`logs/matrix-192/`)

Third topology: 8x RTX 6000D 84G on GB202 + EPYC 9654 (NCCL tuned, ~41 GiB/s
allreduce). Same two-leg recipe as matrix-155/157 (MTP k5 + no-spec, 29 tiers
+ c1 double-run). First boot on this host additionally needed the flashinfer
JIT/autotune warm-up (health window widened to 4000s in the driver script).

Key numbers (output tok/s, k5 vs no-spec): 4K c1 228.1/85.8 (2.66x), 8K c1
279.5/87.3 (3.20x), 32K c1 303.4/87.7 (3.46x), 64K c1 315.5/87.6 (3.60x),
64K c8 843.4/408.9 (2.06x). Crossover unchanged: 1K c128 no-spec wins
(1713.1 vs 1216.4). KV pool 3,302,083 tokens (1M ctx 3.15x). Zero failures
across both legs. Cross-SKU @64K c1 k5: 96G 354.0 > GB202 315.5 > 84G 267.0.
