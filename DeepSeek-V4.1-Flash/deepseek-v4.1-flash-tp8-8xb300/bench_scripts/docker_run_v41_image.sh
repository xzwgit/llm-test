#!/bin/bash
# DeepSeek-V4.1-Flash on 8x B300 SXM6 (SM100), TP8 — 官方镜像
# 模型以本地路径挂载（非 HF repo id）；官方 recipe 的 Blackwell 配置
docker run -d --name v41-b300 --gpus all --privileged --ipc=host -p 8123:8000 \
  -v /data/models:/models \
  -v /root/.cache/flashinfer:/root/.cache/flashinfer \
  -v /root/.cache/vllm:/root/.cache/vllm \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_USE_RUST_FRONTEND=1 \
  -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 \
  vllm/vllm-openai:nightly /models/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --host 0.0.0.0 --api-key abc.12345 \
  --tokenizer-mode deepseek_v41 \
  --tensor-parallel-size 8 \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kv-cache-dtype fp8 \
  --engram-config '{"cpu_offload":true}' \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,18,24,30,36,48,60,72,96,120,144,192,240,288,384,480,576,768,1020,1536,2046,3072,4092,6144,8190]}' \
  --max-cudagraph-capture-size 8190 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --mm-encoder-tp-mode data

# 注: 挂载 /root/.cache/flashinfer 与 /root/.cache/vllm 以保留首次 FlashInfer autotune 结果
# （缓存在容器内，容器删除即丢失；首次 autotune 约 19 分钟）
