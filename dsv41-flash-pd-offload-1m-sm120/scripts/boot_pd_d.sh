#!/bin/bash
# D role: server04 (.154), kv consumer -> P 192.168.200.3
# vision + DSpark dual-pool PD (2026-10-01)
export NCCL_P2P_LEVEL=SYS
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ENGINE_ID=pd-v41-sm120
exec /root/vllm/bin/vllm serve /data/models/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --trust-remote-code --host 0.0.0.0 --port 8123 \
  --tensor-parallel-size 8 \
  --tokenizer-mode deepseek_v41 \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"fp8"}' \
  --kv-cache-dtype fp8 \
  --engram-config '{"cpu_offload":true}' \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --max-model-len 1048576 --max-num-seqs 64 --gpu-memory-utilization 0.92 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --max-cudagraph-capture-size 512 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_ip":"192.168.200.3","kv_port":14579,"kv_parallel_size":1}'
