#!/bin/bash
# KV-offload validation: two identical rounds of 100 distinct 128K prompts (same seed).
# Round1 fills + evicts; Round2 re-sends same prompts -> hit from host pool?
set -u
export OPENAI_API_KEY=<API_KEY>
OUT=/root/kv_offload_test
mkdir -p $OUT
BIN=/root/vllm/bin/vllm
TOK=/data/models/DeepSeek-V4.1-Flash

for RD in 1 2; do
  echo "[$(date +%H:%M:%S)] round $RD start" | tee -a $OUT/progress.log
  $BIN bench serve --host 127.0.0.1 --port 8123 --endpoint /v1/chat/completions \
    --model deepseek-v4.1-flash --tokenizer $TOK \
    --backend openai-chat --dataset-name random \
    --random-input-len 131072 --random-output-len 256 --ignore-eos \
    --max-concurrency 16 --num-prompts 100 --seed 0 \
    > $OUT/round${RD}.log 2>&1
  TT=$(grep -oE "Mean TTFT \(ms\): +[0-9.]+" $OUT/round${RD}.log | tail -1 | grep -oE "[0-9.]+")
  DUR=$(grep -oE "Benchmark duration \(s\): +[0-9.]+" $OUT/round${RD}.log | tail -1 | grep -oE "[0-9.]+")
  IT=$(grep -oE "Total input tokens: +[0-9]+" $OUT/round${RD}.log | tail -1 | grep -oE "[0-9]+")
  echo "[$(date +%H:%M:%S)] round $RD done ttft=${TT}ms dur=${DUR}s in_tok=${IT}" | tee -a $OUT/progress.log
done

echo "== offload-related log lines:" | tee -a $OUT/progress.log
grep -iE "offload|cpu_offload|load.*kv|SimpleCPU" /root/vllm_cpuoffload.log | tail -15 > $OUT/offload_lines.txt 2>&1
cat $OUT/offload_lines.txt | head -8 | tee -a $OUT/progress.log
echo "=== OFFLOAD TEST DONE ===" | tee -a $OUT/progress.log
