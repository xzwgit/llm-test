# DSV4.1-Flash-1017 unified matrix on RTX PRO 6000 96G (SM120)

Same-machine-class counterpart of `dsv41-flash-1017-6000d-sm120`: identical tier labels, ladder, and rules, run on 8x RTX PRO 6000 Blackwell Server Edition 96G (vLLM 0.32.0 + FlashInfer 0.7.2 + #62535, TP8+EP8, util 0.90 uniform, capture 512/640/768, NCCL_P2P_LEVEL=SYS required on this host).

87 random tiers + 9 agent sessions, all zero-failure, single pass (4h09m).

Cross-machine highlights vs RTX 6000D 84G: aggregate tiers +5-13% (1K c128 k3 1966 -> 2217 tok/s), KV pool 17.69M vs 13.2M tokens; gap tracks the memory-bandwidth ratio (1.53 vs 1.30 TB/s = 1.18x), NOT the FP8 compute ratio (906 vs 386 = 2.35x) - serving is bandwidth-priced, see AGENTS.md 6.34.

NOTE: vllm engine log timestamps on this host are LOCAL time (UTC+8) while tier markers are UTC.

Full report: `docs/DSV4.1-Flash-1017-PRO6000-96G-FlashInfer072-DSpark-k345-unified-matrix.html`
