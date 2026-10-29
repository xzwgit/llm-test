#!/bin/bash
# .192: no-speculative unified matrix leg (chained after k7)
# Same ladder as matrix_192.sh minus --speculative-config; VL + Agent included.
OPS=/root/test_ops/k357_192
exec >> $OPS/nospec_192.sh.log 2>&1
set -x

# wait for k7 leg to fully finish (master writes all_done after k7)
for i in $(seq 1 2400); do [ -f $OPS/all_done ] && break; sleep 30; done
[ -f $OPS/all_done ] || { echo "wait for all_done timed out"; exit 1; }
rm -f $OPS/all_done $OPS/nospec_done

export PATH=/usr/local/cuda/bin:$PATH
export OPENAI_API_KEY=abc.12345 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 FLASHINFER_DISABLE_VERSION_CHECK=1
V=/root/vllmenv
MODEL=/data/models/DeepSeek-V4.1-Flash-1017
KLOG=$OPS/matrix_nospec.log
BLOG=$OPS/boot_nospec.log

kill_all () {
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 2
  for p in $(ss -lptnH "sport = :8123" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do kill -9 "$p" 2>/dev/null; done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done
  rm -f /dev/shm/nccl-* /dev/shm/vllm-*; sleep 3
}

health_ok () { curl -s -m 10 http://127.0.0.1:8123/health -H "Authorization: Bearer abc.12345" >/dev/null 2>&1; }

bench () {
  local LOG="$1" BLOG="$2" label="$3" ilen="$4" olen="$5" conc="$6"
  echo "===== $label $(date -u +%H:%M:%S) =====" >> $LOG
  timeout 5400 env OPENAI_API_KEY=abc.12345 $V/bin/vllm bench serve \
    --base-url http://127.0.0.1:8123 --model deepseek-v4.1-flash \
    --tokenizer $MODEL --dataset-name random \
    --random-input-len $ilen --random-output-len $olen --ignore-eos \
    --num-prompts $conc --request-rate $conc >> $LOG 2>&1
  local RC=$?
  [ $RC -ne 0 ] && echo "!!!!! $label BENCH EXIT $RC" >> $LOG
  if ! health_ok; then echo "!!!!! ENGINE DOWN after $label" >> $LOG; return 9; fi
  grep -qE 'Illegal memory access|CUDA error|Engine core failed' $BLOG 2>/dev/null && echo "!!!!! ENGINE ERROR HITS" >> $LOG
  return $RC
}

kill_all
echo "########## NOSPEC_192 BOOT $(date -u) ##########" > $KLOG
echo "BOOT CMD: $V/bin/vllm serve $MODEL --served-model-name deepseek-v4.1-flash --tokenizer-mode deepseek_v41 --trust-remote-code --tensor-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8 --engram-config {cpu_offload:true} --moe-backend auto --compilation-config {cudagraph_mode:PIECEWISE} --max-model-len 1048576 --max-num-seqs 128 --gpu-memory-utilization 0.90 --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --port 8123 --api-key *** [VL ON][NO SPEC]" >> $KLOG

setsid nohup env PATH=/root/vllmenv/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_RUST_FRONTEND=1 VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  NCCL_P2P_LEVEL=SYS FLASHINFER_DISABLE_VERSION_CHECK=1 \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
  $V/bin/vllm serve $MODEL \
  --served-model-name deepseek-v4.1-flash \
  --tokenizer-mode deepseek_v41 --trust-remote-code \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --engram-config '{"cpu_offload":true}' \
  --moe-backend auto \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --max-model-len 1048576 --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
  --port 8123 --api-key abc.12345 \
  > $BLOG 2>&1 < /dev/null &

BOOT_OK=0
for i in $(seq 1 100); do
  sleep 10
  ANS=$(curl -s -m 60 http://127.0.0.1:8123/v1/completions \
    -H "Authorization: Bearer abc.12345" -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4.1-flash","prompt":"17*19=","max_tokens":8,"temperature":0}' \
    2>/dev/null | grep -o '"text":"[^"]*"' | head -1)
  if echo "$ANS" | grep -q 323; then echo "nospec READY after ~$((i*10))s" >> $KLOG; BOOT_OK=1; break; fi
  if grep -qE 'Assertion error|ValueError|Engine core initialization failed|OutOfMemory' $BLOG 2>/dev/null; then
    echo "nospec BOOT FAILED:" >> $KLOG; grep -m5 -E 'Assertion|ValueError|OutOfMemory' $BLOG >> $KLOG; break
  fi
done
if [ "$BOOT_OK" != "1" ]; then echo "FAIL" > $OPS/nospec_done; kill_all; exit 1; fi
grep -E 'KV cache size' $BLOG | tail -2 >> $KLOG
sleep 5

bench $KLOG $BLOG "tier 1K/1K c1 nospec" 1024 1024 1
bench $KLOG $BLOG "tier 1K/1K c1 nospec" 1024 1024 1
for C in 2 4 8 16 32 64 128; do bench $KLOG $BLOG "tier 1K/1K c$C nospec" 1024 1024 $C; done
for C in 1 2 4 8 16 32; do bench $KLOG $BLOG "tier 4K/4K c$C nospec" 4096 4096 $C; done
for C in 1 2 4 8 16 32; do bench $KLOG $BLOG "tier 8K/8K c$C nospec" 8192 8192 $C; done
for C in 1 2 4 8 16; do bench $KLOG $BLOG "tier 32K/32K c$C nospec" 32768 32768 $C; done
for C in 1 2 4 8; do bench $KLOG $BLOG "tier 64K/64K c$C nospec" 65536 65536 $C; done

echo "########## VL E2E $(date -u) ##########" >> $KLOG
$V/bin/python /root/test_ops/vl_e2e.py > $OPS/vl_e2e_nospec.out 2>&1
echo "vl_e2e rc=$?" >> $KLOG; cat $OPS/vl_e2e_nospec.out >> $KLOG

echo "########## NOSPEC AGENT $(date -u) ##########" >> $KLOG
for S in 1 2 3; do
  $V/bin/python /root/test_ops/agent_tools_sim.py http://127.0.0.1:8123 --turns 12 > $OPS/agent_nospec_s$S.out 2>&1
  echo "agent s$S rc=$?" >> $KLOG
done

kill_all
echo "########## NOSPEC_192 DONE $(date -u) ##########" >> $KLOG
echo DONE > $OPS/nospec_done
