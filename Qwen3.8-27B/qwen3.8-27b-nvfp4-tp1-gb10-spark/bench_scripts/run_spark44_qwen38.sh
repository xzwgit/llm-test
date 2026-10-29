#!/bin/bash
# GB10 Spark(.44) unsloth/Qwen3.8-27B-NVFP4 TP1 9档测试(bench在.92远程打)
# 同口径: 3组 x c1/2/4
source ~/.bashrc 2>/dev/null
cd /root/qwen27b-bench
mkdir -p results-spark44-qwen38
rm -f results-spark44-qwen38/ALL_DONE results-spark44-qwen38/ERROR

export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=abc.12345
TARGET=http://10.0.10.44:8123

for i in 1 2 3; do
  curl -s -m 180 $TARGET/v1/chat/completions \
    -H "Authorization: Bearer abc.12345" -H "Content-Type: application/json" \
    -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"ignore_eos":true}' > /dev/null
  echo "=== warmup $i done $(date '+%T') ==="
done

run() {
  local tag=$1 il=$2 ol=$3 n=$4
  echo "=== [$(date '+%F %T')] START $tag il=$il ol=$ol n=$n ==="
  local marker=/tmp/bm38_$tag
  touch "$marker"
  ~/vllm/bin/vllm bench serve \
    --backend openai-chat \
    --base-url $TARGET \
    --endpoint /v1/chat/completions \
    --model Qwen3.8-27B \
    --tokenizer Qwen/Qwen3.8-27B-FP8 \
    --dataset-name random \
    --random-input-len $il \
    --random-output-len $ol \
    --num-prompts $n \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl \
    --metric-percentiles 50,95,99 \
    --save-result --save-detailed \
    > results-spark44-qwen38/${tag}.log 2>&1
  local rc=$?
  local jf=$(find . -maxdepth 1 -name '*.json' -newer "$marker" | head -1)
  if [ -n "$jf" ]; then mv "$jf" results-spark44-qwen38/${tag}.json; fi
  rm -f "$marker"
  echo "=== [$(date '+%F %T')] DONE $tag rc=$rc json=$jf ==="
  if [ $rc -ne 0 ]; then touch results-spark44-qwen38/ERROR; echo "ABORT on $tag"; exit 1; fi
}

for c in 1 2 4; do run in1k-out1k-c$c  1024 1024  $c; done
for c in 1 2 4; do run in4k-out4k-c$c  4096 4096  $c; done
for c in 1 2 4; do run in8k-out8k-c$c  8192 8192  $c; done

touch results-spark44-qwen38/ALL_DONE
echo "=== ALL DONE $(date '+%F %T') ==="
