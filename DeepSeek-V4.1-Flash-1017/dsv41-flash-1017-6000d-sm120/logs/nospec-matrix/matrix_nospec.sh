#!/bin/bash
# .157 无投机完整统一矩阵: DSV4.1-Flash-1017, 不开 DSpark (纯底座基线)
# boot = k345_v3 同款 minus --speculative-config; 29 档 + c1 双跑 + Agent x3 + VL
# 产物: /root/nospec_ops/{master,tiers,boot,agent_s1-3,vl_e2e}.log + {nospec,all}_done
OPS=/root/nospec_ops
mkdir -p $OPS
exec >> $OPS/master.log 2>&1
set -x
export PATH=/usr/local/cuda/bin:$PATH
export OPENAI_API_KEY=abc.12345 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 FLASHINFER_DISABLE_VERSION_CHECK=1
V=/root/v3env
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

boot_engine () {
  local BLOG=$OPS/boot.log
  kill_all
  echo "########## NOSPEC BOOT $(date -u) ##########"
  setsid nohup env PATH=$V/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH CUDA_HOME=/usr/local/cuda \
    VLLM_USE_RUST_FRONTEND=1 VLLM_ENGINE_READY_TIMEOUT_S=7200 \
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

  for i in $(seq 1 110); do
    sleep 10
    ANS=$(curl -s -m 60 http://127.0.0.1:8123/v1/completions \
      -H "Authorization: Bearer abc.12345" -H "Content-Type: application/json" \
      -d '{"model":"deepseek-v4.1-flash","prompt":"17*19=","max_tokens":8,"temperature":0}' \
      2>/dev/null | grep -o '"text":"[^"]*"' | head -1)
    if echo "$ANS" | grep -q 323; then
      echo "NOSPEC READY after ~$((i*10))s"
      grep -E 'KV cache size' $BLOG | tail -2
      return 0
    fi
    if grep -qE 'Assertion error|ValueError|Engine core initialization failed|OutOfMemory' $BLOG 2>/dev/null; then
      echo "NOSPEC BOOT FAILED:"
      grep -m5 -E 'Assertion|ValueError|OutOfMemory|Error' $BLOG
      return 1
    fi
  done
  echo "NOSPEC BOOT TIMEOUT"; return 1
}

run_all () {
  local LLOG=$OPS/tiers.log
  local BLOG=$OPS/boot.log
  if ! boot_engine; then echo "FAIL" > $OPS/nospec_done; return 1; fi
  sleep 5
  echo "===== correctness =====" >> $LLOG
  curl -s -m 120 http://127.0.0.1:8123/v1/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer abc.12345" \
    -d '{"model":"deepseek-v4.1-flash","prompt":"17*19=","max_tokens":16,"temperature":0}' >> $LLOG 2>&1
  echo >> $LLOG
  echo "BOOT CMD: $V/bin/vllm serve $MODEL --served-model-name deepseek-v4.1-flash --tokenizer-mode deepseek_v41 --trust-remote-code --tensor-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8 --engram-config {cpu_offload:true} --moe-backend auto --compilation-config {cudagraph_mode:PIECEWISE} --max-model-len 1048576 --max-num-seqs 128 --gpu-memory-utilization 0.90 --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --port 8123 --api-key *** ENV: VLLM_USE_RUST_FRONTEND=1 NCCL_P2P_LEVEL=SYS 无投机(无 --speculative-config)" >> $LLOG

  # 29 档统一矩阵 (m86/k345-v3 同梯子) + c1 双跑
  bench $LLOG $BLOG "1K/1K c1 run1" 1024 1024 1
  bench $LLOG $BLOG "1K/1K c1 run2" 1024 1024 1
  for C in 2 4 8 16 32 64 128; do bench $LLOG $BLOG "1K/1K c$C" 1024 1024 $C; done
  for C in 1 2 4 8 16 32; do bench $LLOG $BLOG "4K/4K c$C" 4096 4096 $C; done
  for C in 1 2 4 8 16 32; do bench $LLOG $BLOG "8K/8K c$C" 8192 8192 $C; done
  for C in 1 2 4 8 16; do bench $LLOG $BLOG "32K/32K c$C" 32768 32768 $C; done
  for C in 1 2 4 8; do bench $LLOG $BLOG "64K/64K c$C" 65536 65536 $C; done

  # Agent 3 session
  echo "########## AGENT $(date -u) ##########" >> $LLOG
  for S in 1 2 3; do
    $V/bin/python /root/test_ops/agent_tools_sim.py http://127.0.0.1:8123 --turns 12 > $OPS/agent_s$S.out 2>&1
    echo "agent s$S rc=$?" >> $LLOG
  done

  # VL E2E
  echo "########## VL E2E $(date -u) ##########" >> $LLOG
  $V/bin/python /root/test_ops/vl_e2e.py > $OPS/vl_e2e.out 2>&1
  echo "vl_e2e rc=$?" >> $LLOG
  tail -5 $OPS/vl_e2e.out >> $LLOG

  kill_all
  echo "########## NOSPEC DONE $(date -u) ##########" >> $LLOG
  echo DONE > $OPS/nospec_done
  echo DONE > $OPS/all_done
}

echo "########## MASTER START $(date -u) ##########"
run_all
echo "########## MASTER ALL DONE $(date -u) ##########"
