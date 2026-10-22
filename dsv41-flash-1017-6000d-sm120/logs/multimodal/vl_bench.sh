#!/bin/bash
export PATH=/usr/local/cuda/bin:${PATH}
# vl_bench.sh — full multimodal capability bench on one machine:
#   1) official `vllm bench mm-processor` (synthetic random-mm, 3 resolution buckets)
#   2) boot VL server (no --language-model-only) -> E2E: img 512/1024/2048 x c1/c4 + text baseline
# Machine vars injected via sed. Logs: test_ops/mmproc.log, vl_bench.log, vl_boot2.log
set -u
V=/root/vllm-fi02
MODEL=/data/models/DeepSeek-V4.1-Flash-1017
OPS=/root/test_ops
BLOG=$OPS/vl_boot2.log
PLOG=$OPS/vl_bench.log

pkill -9 -f "vllm serve" 2>/dev/null; sleep 2
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done
rm -f /dev/shm/nccl-* /dev/shm/vllm-*; sleep 3
rm -f $OPS/vl_bench.done

echo "########## VL BENCH $(date -u) ##########" > $PLOG

echo "===== [1/2] official mm-processor =====" >> $PLOG
PATH="$V/bin:$PATH" CUDA_VISIBLE_DEVICES=0 timeout 900 $V/bin/vllm bench mm-processor \
  --model $MODEL --dataset-name random-mm --num-prompts 30 \
  --random-input-len 256 --random-output-len 64 \
  --random-mm-base-items-per-request 1 \
  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --random-mm-bucket-config '{(512, 512, 1): 0.5, (1024, 1024, 1): 0.3, (1280, 2048, 1): 0.2}' \
  > $OPS/mmproc.log 2>&1
echo "mm-processor rc=$?" >> $PLOG
grep -E "stage|latency|mean|p50|p99|Processor|Encoder|Total|items|resolution" $OPS/mmproc.log | tail -25 >> $PLOG

echo "===== [2/2] boot VL server =====" >> $PLOG
env PATH="$V/bin:/usr/local/cuda/bin:$PATH" \
  VLLM_USE_RUST_FRONTEND=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  FOO=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  FLASHINFER_DISABLE_VERSION_CHECK=1 HF_HUB_OFFLINE=1 \
  setsid nohup $V/bin/vllm serve $MODEL \
  --served-model-name deepseek-v4.1-flash \
  --tokenizer-mode deepseek_v41 --trust-remote-code \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"fp8"}' \
  --engram-config '{"cpu_offload":true}' \
  --moe-backend auto \
  --compilation-config '{"cudagraph_mode": "FULL"}' \
  --max-model-len 131072 --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3}' \
  --max-cudagraph-capture-size 512 \
  --port 8123 --api-key *** \
  > $BLOG 2>&1 < /dev/null &

READY=0
for i in $(seq 1 120); do
  sleep 10
  ANS=$(curl -s -m 60 http://127.0.0.1:8123/v1/completions \
    -H "Authorization: Bearer ***" -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4.1-flash","prompt":"17*19=","max_tokens":8,"temperature":0}' \
    2>/dev/null | grep -o '"text":"[^"]*"' | head -1)
  if echo "$ANS" | grep -q 323; then
    echo "READY after ~$((i*10))s $(date -u +%H:%M:%S)" >> $PLOG; READY=1; break
  fi
  pgrep -f "vllm serve" >/dev/null || { echo "BOOT PROCESS EXITED" >> $PLOG; break; }
done

if [ "$READY" = "1" ]; then
  echo "===== E2E bench =====" >> $PLOG
  PATH="$V/bin:$PATH" $V/bin/python $OPS/vl_e2e.py >> $PLOG 2>&1
else
  echo "BOOT FAILED:" >> $PLOG; grep -m5 -E 'Error|Assertion|OutOfMemory' $BLOG | tail -5 >> $PLOG
fi

pkill -9 -f "vllm serve" 2>/dev/null; sleep 3
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done
rm -f /dev/shm/nccl-* /dev/shm/vllm-*
pkill -9 -f "[v]llm-rs frontend" 2>/dev/null
echo "########## VL BENCH DONE $(date -u) ##########" >> $PLOG
echo DONE > $OPS/vl_bench.done
