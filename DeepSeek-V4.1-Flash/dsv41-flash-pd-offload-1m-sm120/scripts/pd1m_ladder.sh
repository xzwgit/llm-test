#!/bin/bash
# Extreme long-context concurrency ladder on PD (vision+DSpark dual-pool, 1M max len)
# Every tier's FULL official output is saved. Window timestamps recorded for Prometheus.
set -u
export OPENAI_API_KEY=<API_KEY>
OUT=/root/pd1m
mkdir -p $OUT
BIN=/root/vllm/bin/vllm
TOK=/data/models/DeepSeek-V4.1-Flash

date '+LADDER_START %s' | tee $OUT/window.txt
date '+LADDER_START_H %H:%M:%S' | tee -a $OUT/window.txt

tier(){ # $1=label $2=input_len $3=conc $4=np
  local L=$1 IN=$2 C=$3 NP=$4
  echo "[$(date +%H:%M:%S)] tier $L start (in=$IN c=$C np=$NP)" | tee -a $OUT/progress.log
  date "+TIER_${L}_START %s" >> $OUT/window.txt
  $BIN bench serve --host 127.0.0.1 --port 8130 --endpoint /v1/chat/completions \
    --model deepseek-v4.1-flash --tokenizer $TOK \
    --backend openai-chat --dataset-name random \
    --random-input-len $IN --random-output-len 1024 --ignore-eos \
    --max-concurrency $C --num-prompts $NP \
    > $OUT/tier_${L}.log 2>&1
  echo "[$(date +%H:%M:%S)] tier $L done rc=$?" | tee -a $OUT/progress.log
  date "+TIER_${L}_END %s" >> $OUT/window.txt
  grep -E "Successful requests|Failed requests|Request throughput|Output token throughput|Total token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Mean ITL|Benchmark duration|Total input tokens|Total generated tokens|Acceptance rate|Acceptance length|Drafts:|Draft tokens|Accepted tokens" $OUT/tier_${L}.log | tail -20 > $OUT/tier_${L}.summary
}

# ladder: from c1 upward at each context scale
tier 1M_c1    1048576 1 2
tier 512K_c1   524288 1 2
tier 512K_c2   524288 2 4
tier 256K_c1   262144 1 3
tier 256K_c2   262144 2 6
tier 256K_c4   262144 4 8
tier 128K_c1   131072 1 4
tier 128K_c2   131072 2 8
tier 128K_c4   131072 4 12
tier 128K_c8   131072 8 16

date '+LADDER_END %s' | tee -a $OUT/window.txt
date '+LADDER_END_H %H:%M:%S' | tee -a $OUT/window.txt
echo "=== ALL DONE ===" | tee -a $OUT/progress.log
ls -la $OUT/
