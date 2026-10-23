# SM120 (RTX PRO 6000 / 6000D) · DeepSeek-V4.1-Flash on vLLM 0.32.0 — dual patch snapshot (v2)

Run DSV4.1-Flash (incl. 1017) with **DSpark speculative decoding + FULL cudagraphs** on SM120
using the **stock vLLM 0.32.0 wheel + FlashInfer 0.7.2**. Two upstream patches are required;
this snapshot ships them as drop-in files for the 0.32.0 tree.

**v2 (2026-10-23): the graph-admission half is now the port of vllm#59986** (builder-layer,
the blessed fix). v1 shipped our runtime guard vllm#62535, which was closed as superseded by
#59986 after both arch legs validated — v1 files remain in git history (commit 9c8970a).

## What each patch does

| Patch | Files (repo-relative) | Purpose |
|---|---|---|
| [#60762](https://github.com/vllm-project/vllm/pull/60762) | `indexer.py`, `flashinfer_sparse.py` | SM120 block-size negotiation must resolve to 64 (DeepGEMM/FlashInfer SM120 kernels only accept block_kv ∈ {32,64}); stock 0.32.0 declares [128] and dies at boot with the `attention.hpp:411` assertion |
| [#59986](https://github.com/vllm-project/vllm/pull/59986) | `compressor.py`, `sparse_mla.py`, `flashmla.py`, `flashinfer_sparse.py` | Five DSV4.1 sparse metadata builders declare `AttentionCGSupport.ALWAYS`, letting prefill-bearing batches enter FULL decode-only cudagraphs -> illegal memory access / garbage output (vllm#59970, vllm#62417). Downgraded to `UNIFORM_BATCH` + varlen decode bound, the engine resolves `cudagraph_mode=FULL -> FULL_DECODE_ONLY` at boot and the failure class is gone |

Both are needed; neither subsumes the other (independent axes). With only #60762 the engine
boots but the first prefill batch entering a FULL graph kills it / corrupts DSpark chat output.

## Usage

Copy each file over its package-relative path (mapping in the table above; `flashinfer_sparse.py`
and `flashmla.py` live under `vllm/models/deepseek_v41/nvidia/`, `indexer.py` under
`vllm/v1/attention/backends/mla/`), then boot with FULL + DSpark — see `verification/evidence.txt`
for the exact command line and expected engine log line:

```
CUDAGraphMode.FULL is not supported with ... (support: AttentionCGSupport.UNIFORM_BATCH);
setting cudagraph_mode=FULL_DECODE_ONLY
```

`dual-patch-vs-stock-v0.32.0.diff` is the combined minimal diff (all five files vs pristine stock)
if you prefer applying a patch over copying files.

## Validation

- SM120 leg (8x RTX PRO 6000D 84G, .158): corruption probes 3/3, random 1Kx1K c32 burst 32/32,
  zero engine errors — details and RECORD hash audit in `verification/evidence.txt`
  (only the 5 patched files differ from the pristine wheel; no other residue).
- B300 leg (2x, @tomasruizt): GSM8K 0.955, all probes clean — see the PR thread.
- Perf note: steady-state throughput/TPOT/ITL match the runtime-guard form within noise
  (same execution path; only the admission decision moved to boot-time).

## Notes

- Ported from #59986 head `d29af5185`. Annotations use quoted forward refs (`"VllmConfig"`)
  where upstream adds imports — functionally identical, keeps the diff minimal.
- Validated on the PyPI wheel tree. Conda-forge 0.32.0 builds have minor API drift
  (e.g. `BatchExecutionDescriptor` lacks `is_full`) — re-check the anchors before use there.
- Default PIECEWISE mode needs none of this; this snapshot is for FULL + DSpark serving.
