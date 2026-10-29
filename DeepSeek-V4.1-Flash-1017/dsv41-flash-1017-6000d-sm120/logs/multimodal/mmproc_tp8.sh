#!/bin/bash
export PATH=/root/vllm-fi02/bin:/usr/local/cuda/bin:${PATH}
# mmproc_tp8.sh — official vllm bench mm-processor with TP8 (TP=1 cannot fit the 476G model)
set -u
OPS=/root/test_ops
rm -f $OPS/mmproc_tp8.done
timeout 900 vllm bench mm-processor \
  --model /data/models/DeepSeek-V4.1-Flash-1017 \
  --dataset-name random-mm \
  --num-prompts 30 \
  --random-input-len 256 \
  --random-output-len 64 \
  --random-mm-base-items-per-request 1 \
  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --random-mm-bucket-config '{(512, 512, 1): 0.5, (1024, 1024, 1): 0.3, (1280, 2048, 1): 0.2}' \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  > $OPS/mmproc_tp8.log 2>&1
echo "rc=$?" >> $OPS/mmproc_tp8.log
echo DONE > $OPS/mmproc_tp8.done
