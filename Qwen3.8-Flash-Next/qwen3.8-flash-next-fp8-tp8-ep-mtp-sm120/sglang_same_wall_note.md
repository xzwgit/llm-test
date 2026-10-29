SGLang 0.5.21 + torch 2.13.0+cu132, same box (8x RTX PRO 6000), 2026-10-09.
Pure TP8 (no EP) load-time rejection, identical wording to vLLM mainline:

  File "/root/sglang31/lib/python3.12/site-packages/sglang/srt/layers/quantization/fp8.py", line 1650, in create_weights
  File "/root/sglang31/lib/python3.12/site-packages/sglang/srt/layers/quantization/fp8.py", line 1403, in create_fp8_moe_weight_
    raise ValueError(
ValueError: The output_size of gate's and up's weight = 80 is not divisible by weight quantization block_n = 128.
  File "/root/sglang31/lib/python3.12/site-packages/sglang/srt/layers/quantization/fp8.py", line 1650, in create_weights
  File "/root/sglang31/lib/python3.12/site-packages/sglang/srt/layers/quantization/fp8.py", line 1403, in create_fp8_moe_weight_
    raise ValueError(
ValueError: The output_size of gate's and up's weight = 80 is not divisible by weight quantization block_n = 128.
  File "/root/sglang31/lib/p

Architecture itself IS supported (Qwen4ExpForConditionalGeneration dispatches; 8 workers spawn; fails at weight creation, fp8.py:1403 Fp8MoEMethod.create_weights — before reading any weights). No refine/requant fallback in SGLang (vLLM 0.31.1 main at least refines blocks at TP4: 160=5x32).
Full log: sglang_tp8noep_fail.log.