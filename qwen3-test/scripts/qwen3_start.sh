#!/bin/bash
# Qwen3.6-27B-FP8 启动脚本 (修正前后台 Rust frontend 不一致 bug)
# 单卡模式: 直接 bash qwen3_start.sh
# 多卡模式: 改下面 TP_SIZE 和 GPU_IDS

MODEL_NAME="Qwen/Qwen3.6-27B-FP8"
SERVED_MODEL_NAME="Qwen3.6-27B"
PORT=8123

# ===== GPU 配置 (按需改) =====
GPU_IDS="0"                  # 单卡: "0" | 4卡: "0,1,2,3" | 8卡: "0,1,2,3,4,5,6,7"
TP_SIZE=1                    # 单卡:1 | 4卡:4 | 8卡:8
# ==============================

MAX_MODEL_LEN=131072         # 降到128K (27B单卡256K偏紧, 128K更稳)
GPU_MEM_UTIL=0.92            # 降到0.92 (留余量给激活, 防长输出OOM)

source ~/vllm/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU_IDS}
# ★ 修正: 统一 Rust frontend 开关 (原脚本前台0后台1是bug)
export VLLM_USE_RUST_FRONTEND=0

vllm serve ${MODEL_NAME} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --trust-remote-code \
  --port ${PORT} \
  --host 0.0.0.0 \
  --tensor-parallel-size ${TP_SIZE} \
  --dtype auto \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --log-stats
