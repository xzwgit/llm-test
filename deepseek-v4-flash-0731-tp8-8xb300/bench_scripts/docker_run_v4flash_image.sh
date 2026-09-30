#!/bin/bash
# DeepSeek-V4-Flash-0731 on 8x B300 SXM6 (SM100), TP8 — 官方镜像 + V4-Flash recipe 配置
docker run -d --name v4flash-b300 --gpus all --privileged --ipc=host -p 8123:8000 \
  -v /data/models:/models \
  -v /root/.cache/flashinfer:/root/.cache/flashinfer \
  -v /root/.cache/vllm:/root/.cache/vllm \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_USE_RUST_FRONTEND=1 \
  -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 \
  vllm/vllm-openai:nightly /models/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash-0731 \
  --host 0.0.0.0 --api-key abc.12345 \
  --kv-cache-dtype fp8 --block-size 256 --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --attention_config.indexer_kv_dtype fp8 --moe-backend auto \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --max-model-len 131072 --max-num-seqs 256
