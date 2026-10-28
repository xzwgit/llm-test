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
