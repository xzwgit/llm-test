#!/bin/bash
# P/D ratio capacity measurement:
#  A) P prefill capacity: direct to P (.153:8123), long-in/short-out, sweep concurrency
#  B) D decode capacity:  via router (8130), short-in/long-out, high concurrency
set -u
export OPENAI_API_KEY=<API_KEY>
OUT=/root/pd_ratio
mkdir -p $OUT
BIN=/root/vllm/bin/vllm
TOK=/data/models/DeepSeek-V4.1-Flash

date '+RATIO_START %s' | tee $OUT/window.txt

echo "===== A) P prefill capacity (direct .153:8123) ====="
for IN in 32768 131072; do
  for C in 4 16 32; do
    L=prefill_${IN}_c${C}
    echo "[$(date +%H:%M:%S)] $L start" | tee -a $OUT/progress.log
    $BIN bench serve --host <P_NODE_MGMT_IP> --port 8123 --endpoint /v1/chat/completions \
      --model deepseek-v4.1-flash --tokenizer $TOK \
      --backend openai-chat --dataset-name random \
      --random-input-len $IN --random-output-len 128 --ignore-eos \
      --max-concurrency $C --num-prompts $((C*3)) \
      > $OUT/${L}.log 2>&1
    OK=$(grep -oE "Successful requests: +[0-9]+" $OUT/${L}.log | tail -1 | grep -oE "[0-9]+")
    IT=$(grep -oE "Total token throughput \(tok/s\): +[0-9.]+" $OUT/${L}.log | tail -1 | grep -oE "[0-9.]+")
    echo "[$(date +%H:%M:%S)] $L ok=$OK total_tput=${IT:-?}" | tee -a $OUT/progress.log
  done
done

echo "===== B) D decode capacity (via router, 256-in/8192-out) ====="
for C in 32 64; do
  L=decode_c${C}
  echo "[$(date +%H:%M:%S)] $L start" | tee -a $OUT/progress.log
  $BIN bench serve --host 127.0.0.1 --port 8130 --endpoint /v1/chat/completions \
    --model deepseek-v4.1-flash --tokenizer $TOK \
    --backend openai-chat --dataset-name random \
    --random-input-len 256 --random-output-len 8192 --ignore-eos \
    --max-concurrency $C --num-prompts $((C/2)) \
    > $OUT/${L}.log 2>&1
  OK=$(grep -oE "Successful requests: +[0-9]+" $OUT/${L}.log | tail -1 | grep -oE "[0-9]+")
  OT=$(grep -oE "Output token throughput \(tok/s\): +[0-9.]+" $OUT/${L}.log | tail -1 | grep -oE "[0-9.]+")
  TP=$(grep -oE "Mean TPOT \(ms\): +[0-9.]+" $OUT/${L}.log | tail -1 | grep -oE "[0-9.]+")
  echo "[$(date +%H:%M:%S)] $L ok=$OK out_tput=${OT:-?} tpot=${TP:-?}" | tee -a $OUT/progress.log
done

date '+RATIO_END %s' | tee -a $OUT/window.txt
echo "=== RATIO BENCH DONE ===" | tee -a $OUT/progress.log
