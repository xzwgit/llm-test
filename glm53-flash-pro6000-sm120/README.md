# GLM-5.3-Flash on RTX PRO 6000 (96G, SM120) x8 TP8

vLLM nightly (0.33.0rc1.dev174) + FlashInfer 0.7.2.post1 + FIXB patch
(vLLM_BREAKABLE_CAPTURE_ATTN=1, PR vllm-project/vllm#64266). Official MTP
speculative decoding (num_speculative_tokens=5) vs no-spec baseline.

Stack provenance: GLM MoE runs on DEEPGEMM FP8 (PR #63533 domain) and
attention on FLASHINFER_MLA_SPARSE_SM120 (PR #60762 domain).

## 2026-10-28 · two-leg 29-tier matrix (`logs/matrix-155/`)

| Path | Content |
|---|---|
| `logs/matrix-155/matrix_k5.log` | Raw bench output, MTP k5 leg, 29 tiers + c1 double-run |
| `logs/matrix-155/matrix_nospec.log` | Raw bench output, no-spec leg, 29 tiers + c1 double-run |
| `logs/matrix-155/spec_k5.log` | Engine-side SpecDecoding metrics (300 windows) |
| `logs/matrix-155/master.log` | Driver trace incl. BOOT CMD lines |
| `logs/matrix-155/boot_k{5}.log`, `boot_nospec.log` | Server boot logs (host addr sanitized) |
| `logs/matrix-155/glm_matrix.sh` | Driver script |
| `logs/matrix-155/tiers.tsv` | Parsed per-tier throughput/latency table |
| `logs/matrix-155/acceptance_k5.tsv` | Parsed per-tier acceptance (bench window sum) |

Key numbers (output tok/s, k5 vs nospec):
- Single stream c1: 1K 163.8/81.8 (2.00x), 4K 255.4/89.2 (2.86x), 8K 330.7/88.3 (3.74x), 32K 334.1/89.3 (3.74x), 64K 354.0/89.1 (3.97x)
- High concurrency: 1K c128 nospec wins (1932.9 vs 1465.1); 4K+ bands k5 wins everywhere (32K c16 2.06x, 64K c8 1.96x)
- Acceptance peaks at long context: 64K c1 98.4%/5.92; lowest at 1K band (2.2-3.6)
- Zero failures across both legs; c1 double-run cold-start delta +5.4% on k5
