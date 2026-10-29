#!/usr/bin/env bash
# DeepSeek-V4-Pro DP2 主节点启动脚本 (server01, 10.10.3.15)
# 用法: bash start_server01.sh
set -euo pipefail
source ~/vllm/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export VLLM_USE_RUST_FRONTEND=1

vllm serve deepseek-ai/DeepSeek-V4-Pro \
  --served-model-name DeepSeek-V4-Pro \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-address 10.10.3.15 \
  --data-parallel-rpc-port 13345 \
  --moe-backend marlin \
  --linear-backend deep_gemm \
  --gpu-memory-utilization 0.95 \
  --max-model-len 524288 \
  --max-num-batched-tokens 8192 \
  --compilation-config '{}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --port 8123 \
  --api-key abc.12345
