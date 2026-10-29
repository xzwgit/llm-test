#!/bin/bash
# 压测编排：1K in / 1K out，c1 与 c4。用法: bench_1k.sh <base_url> <model> <tokenizer> <outdir>
BASE=$1; MODEL=$2; TOK=$3; OUT=$4
export HF_HUB_OFFLINE=1 OPENAI_API_KEY=abc.12345
mkdir -p "$OUT"; cd "$OUT"
# 同形状预热，避免首次形状 JIT 造成的 TTFT 假象
for i in 1 2; do
  curl -s -m 300 $BASE/v1/chat/completions -H "Authorization: Bearer abc.12345" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":32,\"ignore_eos\":true}" >/dev/null
done
for c in 1 4; do
  echo "=== [$(date '+%T')] 1k-c$c ==="
  vllm bench serve --backend openai-chat --base-url $BASE \
    --header "Authorization=Bearer abc.12345" --endpoint /v1/chat/completions \
    --model $MODEL --tokenizer $TOK \
    --dataset-name random --random-input-len 1024 --random-output-len 1024 \
    --num-prompts $c --ignore-eos \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 \
    --save-result > 1k-c$c.log 2>&1
  grep -E 'Successful|Failed|Total input|Total generated|Output token throughput|Mean TTFT|Mean TPOT' 1k-c$c.log
done
