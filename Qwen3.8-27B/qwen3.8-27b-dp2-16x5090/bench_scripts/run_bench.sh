#!/bin/bash
# Qwen3.8-27B DP=2 26档压测 (vllm bench serve 官方口径)
export no_proxy=localhost,127.0.0.1
VLLM=/root/vllm/bin/vllm
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ | head -1)
DIR=/root/qwen27b-bench/results
mkdir -p $DIR
AUTH='Authorization=Bearer <API-KEY>'
PCT='--percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99'

# 预热一发(不计入)
timeout 300 $VLLM bench serve --backend openai-chat --host 127.0.0.1 --port 8123 \
  --endpoint /v1/chat/completions --header "$AUTH" --model Qwen3.8-27B --tokenizer "$SNAP" \
  --dataset-name random --random-input-len 1024 --random-output-len 256 --num-prompts 4 \
  --ignore-eos > /dev/null 2>&1

run_level () {  # $1=tag $2=in $3=out $4=concurrency
  echo "[$(date +%H:%M:%S)] START $1 (in=$2 out=$3 c=$4)"
  timeout 3600 $VLLM bench serve --backend openai-chat --host 127.0.0.1 --port 8123 \
    --endpoint /v1/chat/completions --header "$AUTH" --model Qwen3.8-27B --tokenizer "$SNAP" \
    --dataset-name random --random-input-len $2 --random-output-len $3 --random-range-ratio 0 \
    --num-prompts $4 --ignore-eos $PCT \
    --save-result --result-dir $DIR --result-filename "$1.json" \
    > $DIR/$1.log 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $1 rc=$?"
}

for c in 1 2 4 16 32 64; do run_level "1k1k-c$c" 1024 1024 $c; done
for c in 1 2 4 16 32;     do run_level "4k4k-c$c" 4096 4096 $c; done
for c in 1 2 4 16 32;     do run_level "8k8k-c$c" 8192 8192 $c; done
for c in 1 2 4 16 32;     do run_level "4k32k-c$c" 4096 32768 $c; done
for c in 1 2 4 16 32;     do run_level "8k32k-c$c" 8192 32768 $c; done
echo "ALL DONE $(date +%H:%M:%S)"
