#!/bin/bash
# .192: k=3/5/7 统一矩阵 (GB202 6000D 84G, vllmenv=dev463+全补丁, NCCL 41GiB/s 调优机)
# 同 m86 梯子(29档/k + c1 双跑) + VL + Agent
OPS=/root/test_ops/k357_192
exec >> $OPS/master.log 2>&1
set -x
export PATH=/usr/local/cuda/bin:$PATH
export OPENAI_API_KEY=abc.12345 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 FLASHINFER_DISABLE_VERSION_CHECK=1
V=/root/vllmenv
MODEL=/data/models/DeepSeek-V4.1-Flash-1017

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

boot_k () {
  local K="$1"
  local CAP=$(( (128 * (K + 1) + 255) / 256 * 256 ))
  [ $CAP -lt 512 ] && CAP=512
  local BLOG=$OPS/boot_k$K.log
  local KLOG=$OPS/matrix_k$K.log
  kill_all
  echo "########## K357_192 k=$K BOOT (align patch, nightly) $(date -u) ##########" > $KLOG
  echo "BOOT CMD: $V/bin/vllm serve $MODEL --served-model-name deepseek-v4.1-flash --tokenizer-mode deepseek_v41 --trust-remote-code --tensor-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8 --engram-config {cpu_offload:true} --moe-backend auto --compilation-config {cudagraph_mode:PIECEWISE} --max-model-len 1048576 --max-num-seqs 128 --gpu-memory-utilization 0.90 --speculative-config {method:dspark,num_speculative_tokens:$K} --max-cudagraph-capture-size $CAP --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --port 8123 --api-key *** [VL ON]" >> $KLOG

  setsid nohup env PATH=/root/v3env/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH CUDA_HOME=/usr/local/cuda VLLM_USE_RUST_FRONTEND=1 VLLM_ENGINE_READY_TIMEOUT_S=7200 \
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
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$K}" \
    --max-cudagraph-capture-size $CAP \
    --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
    --port 8123 --api-key abc.12345 \
    > $BLOG 2>&1 < /dev/null &

  for i in $(seq 1 100); do
    sleep 10
    ANS=$(curl -s -m 60 http://127.0.0.1:8123/v1/completions \
      -H "Authorization: Bearer abc.12345" -H "Content-Type: application/json" \
      -d '{"model":"deepseek-v4.1-flash","prompt":"17*19=","max_tokens":8,"temperature":0}' \
      2>/dev/null | grep -o '"text":"[^"]*"' | head -1)
    if echo "$ANS" | grep -q 323; then
      echo "k=$K READY after ~$((i*10))s" >> $KLOG
      grep -E 'KV cache size' $BLOG | tail -2 >> $KLOG
      return 0
    fi
    if grep -qE 'Assertion error|ValueError|Engine core initialization failed|OutOfMemory' $BLOG 2>/dev/null; then
      echo "k=$K BOOT FAILED:" >> $KLOG
      grep -m5 -E 'Assertion|ValueError|OutOfMemory' $BLOG >> $KLOG
      return 1
    fi
  done
  echo "k=$K BOOT TIMEOUT" >> $KLOG; return 1
}

run_k () {
  local K="$1"
  local KLOG=$OPS/matrix_k$K.log
  local BLOG=$OPS/boot_k$K.log
  if ! boot_k $K; then echo "FAIL boot" > $OPS/k${K}_done; return; fi
  sleep 5
  echo "===== correctness k=$K =====" >> $KLOG
  curl -s -m 120 http://127.0.0.1:8123/v1/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer abc.12345" \
    -d '{"model":"deepseek-v4.1-flash","prompt":"17*19=","max_tokens":16,"temperature":0}' >> $KLOG 2>&1
  echo >> $KLOG

  # c1 双跑
  bench $KLOG $BLOG "tier 1K/1K c1 k=$K" 1024 1024 1
  bench $KLOG $BLOG "tier 1K/1K c1 k=$K" 1024 1024 1
  for C in 2 4 8 16 32 64 128; do bench $KLOG $BLOG "tier 1K/1K c$C k=$K" 1024 1024 $C; done
  for C in 1 2 4 8 16 32; do bench $KLOG $BLOG "tier 4K/4K c$C k=$K" 4096 4096 $C; done
  for C in 1 2 4 8 16 32; do bench $KLOG $BLOG "tier 8K/8K c$C k=$K" 8192 8192 $C; done
  for C in 1 2 4 8 16; do bench $KLOG $BLOG "tier 32K/32K c$C k=$K" 32768 32768 $C; done
  for C in 1 2 4 8; do bench $KLOG $BLOG "tier 64K/64K c$C k=$K" 65536 65536 $C; done

  # VL (k=3 boot)
  if [ "$K" = "3" ]; then
    echo "########## VL E2E $(date -u) ##########" >> $KLOG
    $V/bin/python /root/test_ops/vl_e2e.py > $OPS/vl_e2e.out 2>&1
    echo "vl_e2e rc=$?" >> $KLOG
    cat $OPS/vl_e2e.out >> $KLOG
  fi

  # Agent 3 session
  echo "########## k=$K AGENT $(date -u) ##########" >> $KLOG
  for S in 1 2 3; do
    $V/bin/python /root/test_ops/agent_tools_sim.py http://127.0.0.1:8123 --turns 12 > $OPS/agent_k${K}_s$S.out 2>&1
    echo "agent s$S rc=$?" >> $KLOG
  done
  grep -E 'SpecDecoding metrics' $BLOG | tail -150 > $OPS/spec_k$K.log

  kill_all
  echo "########## K357_192 k=$K DONE $(date -u) ##########" >> $KLOG
  echo DONE > $OPS/k${K}_done
}

echo "########## MASTER START $(date -u) ##########"
for K in ${KLIST:-3 5 7}; do run_k $K; done
kill_all
echo "########## MASTER ALL DONE $(date -u) ##########"
echo DONE > $OPS/all_done
