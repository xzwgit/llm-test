#!/bin/bash
# GB10(.44) 组1: FP8主模型 + DFlash2(num_spec=7) 9档
source ~/.bashrc 2>/dev/null
cd /root/qwen27b-bench
mkdir -p results-spark44-dflash7-fp8
rm -f results-spark44-dflash7-fp8/ALL_DONE results-spark44-dflash7-fp8/ERROR

export HF_HUB_OFFLINE=1
export OPENAI_API_KEY=abc.12345
TARGET=http://10.0.10.44:8123

echo "=== waiting for service ready ..."
for i in $(seq 1 60); do
  curl -s -m 5 $TARGET/v1/models -H "Authorization: Bearer abc.12345" | grep -q Qwen3.8-27B && break
  sleep 15
done
echo "=== service ready, warming up $(date '+%F %T')"

for i in 1 2 3; do
  curl -s -m 300 $TARGET/v1/chat/completions \
    -H "Authorization: Bearer abc.12345" -H "Content-Type: application/json" \
    -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"ignore_eos":true}' > /dev/null
  echo "=== warmup $i done $(date '+%T') ==="
done

run() {
  local tag=$1 il=$2 ol=$3 n=$4
  echo "=== [$(date '+%F %T')] START $tag il=$il ol=$ol n=$n ==="
  local marker=/tmp/bmdf7f_$tag
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
    > results-spark44-dflash7-fp8/${tag}.log 2>&1
  local rc=$?
  local jf=$(find . -maxdepth 1 -name '*.json' -newer "$marker" | head -1)
  if [ -n "$jf" ]; then mv "$jf" results-spark44-dflash7-fp8/${tag}.json; fi
  rm -f "$marker"
  echo "=== [$(date '+%F %T')] DONE $tag rc=$rc json=$jf ==="
  if [ $rc -ne 0 ]; then touch results-spark44-dflash7-fp8/ERROR; echo "ABORT on $tag"; exit 1; fi
}

for c in 1 2 4; do run in1k-out1k-c$c  1024 1024  $c; done
for c in 1 2 4; do run in4k-out4k-c$c  4096 4096  $c; done
for c in 1 2 4; do run in8k-out8k-c$c  8192 8192  $c; done

touch results-spark44-dflash7-fp8/ALL_DONE
echo "=== ALL DONE $(date '+%F %T') ==="
