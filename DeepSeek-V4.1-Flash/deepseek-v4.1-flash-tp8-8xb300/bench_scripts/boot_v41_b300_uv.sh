#!/bin/bash
# DeepSeek-V4.1-Flash on 8x B300 SXM6 (SM100), TP8 — 本地 uv 环境
# 环境: /root/vllm = vLLM 0.30.1rc1.dev412 + flashinfer 0.7.0.post1 + DeepGEMM 6901431(源码编译)
# 必须显式指定 moe_backend=deep_gemm: 该组合在 SM100 上默认选 flashinfer_trtllm,
# 其 JIT 编译失败(TRT-LLM 头文件重复定义) -> 改走 DeepGEMM
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
ulimit -n 1048576
exec /root/vllm/bin/vllm serve /data/models/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --trust-remote-code --host 0.0.0.0 --port 8123 --api-key abc.12345 \
  --tokenizer-mode deepseek_v41 \
  --tensor-parallel-size 8 \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kv-cache-dtype fp8 \
  --engram-config '{"cpu_offload":true}' \
  --kernel-config '{"moe_backend":"deep_gemm"}' \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046,3072,4092,6144,8190]}' \
  --max-cudagraph-capture-size 8190 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --mm-encoder-tp-mode data
