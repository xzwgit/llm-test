# Release 0.32.0 dual-patch snapshot (vLLM 0.32.0 + FlashInfer 0.7.2 on SM120)

Source: the conda env on server07 (RTX 6000D 84G) that validated the full chain
(stock crash -> +block declaration -> +dispatch guard -> clean boot + DSpark output).
These 7 files reproduce a working SM120 environment on ANY stock vLLM 0.32.0 tree
(both PyPI and conda builds):

- `indexer.py` / `flashinfer_sparse.py` = the SM120 -> [64] block-size declaration
  (= upstream PR vllm-project/vllm#60762; `.stock_bak` files are the untouched originals,
  kept as evidence of the empty-intersection failure on stock)
- `model_runner.py` / `dp_utils.py` / `cudagraph_utils.py` = the allow_full dispatch
  guard (= upstream PR vllm-project/vllm#62535)

Both patches are required: #60762 fixes the boot-time block negotiation
(stock resolves block 128, the DeepGEMM paged-MQA-logits kernel asserts block_kv in {32,64}),
#62535 keeps prefill-bearing batches out of FULL decode graphs (without it the engine
boots but the first request dies - `'BatchExecutionDescriptor' object has no attribute 'is_full'`
style failures come from porting the guard with the wrong attribute name).

Porting note (conda vs PyPI): the conda build's BatchExecutionDescriptor lacks `is_full`;
the guard must use `desc.cg_mode == CUDAGraphMode.FULL` (as these files do). PR heads are
main-based and already use the cg_mode form.

Apply: copy the 5 non-.bak files over the same paths in
`<env>/lib/python3.12/site-packages/vllm/...`. Full recipe: AGENTS.md 6.36.
