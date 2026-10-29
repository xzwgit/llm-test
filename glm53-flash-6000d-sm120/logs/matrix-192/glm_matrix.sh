#!/bin/bash
# .192: GLM-5.3-Flash matrix, two legs: MTP k5 (official) -> no-spec
# 29-tier ladder + c1 double-run, official vllm bench serve; per-tier health check.
# .192 adaptation: CUDA toolkit at /usr/local/cuda (no cuda-13.0 on this host).
OPS=/root/glm_matrix
exec >> $OPS/master.log 2>&1
set -x
export PATH=/usr/local/cuda/bin:$PATH
export OPENAI_API_KEY=abc.12345 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 FLASHINFER_DISABLE_VERSION_CHECK=1
V=/root/fixbenv
MODEL=/data/models/GLM-5.3-Flash

kill_all () {
  pkill -9 -f "[v]llm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 2
  for p in $(ss -lptnH "sport = :8123" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do kill -9 "$p" 2>/dev/null; done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done
  rm -f /dev/shm/nccl-* /dev/shm/vllm-*; sleep 5
}

health_ok () { curl -s -m 10 http://127.0.0.1:8123/health -H "Authorization: Bearer abc.12345" >/dev/null 2>&1; }

bench () {
  local LOG="$1" BLOG="$2" label="$3" ilen="$4" olen="$5" conc="$6"
  echo "===== $label $(date -u +%H:%M:%S) =====" >> $LOG
  timeout 5400 env OPENAI_API_KEY=abc.12345 $V/bin/vllm bench serve \
    --backend openai --base-url http://127.0.0.1:8123 --model glm-5.3-flash \
    --tokenizer $MODEL --endpoint /v1/completions --dataset-name random \
    --random-input-len $ilen --random-output-len $olen --ignore-eos \
    --num-prompts $conc --request-rate $conc >> $LOG 2>&1
  local RC=$?
  [ $RC -ne 0 ] && echo "!!!!! $label BENCH EXIT $RC" >> $LOG
  if ! health_ok; then echo "!!!!! ENGINE DOWN after $label" >> $LOG; return 9; fi
  grep -qE 'Illegal memory access|CUDA error|Engine core failed' $BLOG 2>/dev/null && echo "!!!!! ENGINE ERROR HITS" >> $LOG
  return $RC
}

boot_leg () {
  local LEG="$1"
  local SPECARGS=""
  local TAGTXT="NOSPEC"
  if [ "$LEG" = "k5" ]; then
    SPECARGS="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":5}"
    TAGTXT="MTP k5"
  fi
  local BLOG=$OPS/boot_${LEG}.log
  local LLOG=$OPS/matrix_${LEG}.log
  kill_all
  echo "########## GLM_MATRIX ${TAGTXT} BOOT $(date -u) ##########" > $LLOG
  echo "BOOT CMD: $V/bin/vllm serve $MODEL --served-model-name glm-5.3-flash --tensor-parallel-size 8 --gpu-memory-utilization 0.90 --port 8123 --api-key *** --trust-remote-code $SPECARGS" >> $LLOG
  echo "ENV: VLLM_BREAKABLE_CAPTURE_ATTN=1 VLLM_USE_RUST_FRONTEND=1 NCCL_P2P_LEVEL=SYS HF_HUB_OFFLINE=1" >> $LLOG

  setsid nohup env PATH=/root/fixbenv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \
    VLLM_USE_RUST_FRONTEND=1 VLLM_ENGINE_READY_TIMEOUT_S=7200 NCCL_P2P_LEVEL=SYS \
    VLLM_BREAKABLE_CAPTURE_ATTN=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
    $V/bin/vllm serve $MODEL \
    --served-model-name glm-5.3-flash \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 \
    --port 8123 --api-key abc.12345 \
    --trust-remote-code $SPECARGS \
    > $BLOG 2>&1 < /dev/null &

  local BOOT_OK=0
  # 400x10s window: first boot on a fresh host runs flashinfer JIT compile +
  # sparse_mla_sm120 autotune (GPU-idle CPU phase) which can exceed 1000s.
  for i in $(seq 1 400); do
    sleep 10
    if health_ok; then
      # settle + KV line
      sleep 20
      echo "${TAGTXT} READY after ~$((i*10+20))s" >> $LLOG
      grep -E 'KV cache size|Maximum concurrency' $BLOG | tail -2 >> $LLOG
      BOOT_OK=1; break
    fi
    if grep -qE 'Assertion error|ValueError|Engine core initialization failed|OutOfMemory|terminate called|failed to be inspected' $BLOG 2>/dev/null; then
      echo "${TAGTXT} BOOT FAILED:" >> $LLOG
      grep -m5 -E 'Assertion|ValueError|OutOfMemory|terminate called|Error|failed to be inspected' $BLOG >> $LLOG
      break
    fi
  done
  if [ "$BOOT_OK" != "1" ]; then echo "FAIL" > $OPS/${LEG}_done; kill_all; return 1; fi
  sleep 10

  # correctness probe (thinking model: full-instruction prompts, no seed)
  echo "===== correctness ${TAGTXT} =====" >> $LLOG
  curl -s -m 120 http://127.0.0.1:8123/v1/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer abc.12345" \
    -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"请计算下面的算式并只输出最终结果，不要输出任何其他内容: 17 × 19 ="}],"temperature":0,"max_tokens":512}' >> $LLOG 2>&1
  echo >> $LLOG

  bench $LLOG $BLOG "tier 1K/1K c1 ${LEG}" 1024 1024 1
  bench $LLOG $BLOG "tier 1K/1K c1 ${LEG}" 1024 1024 1
  for C in 2 4 8 16 32 64 128; do bench $LLOG $BLOG "tier 1K/1K c$C ${LEG}" 1024 1024 $C; done
  for C in 1 2 4 8 16 32; do bench $LLOG $BLOG "tier 4K/4K c$C ${LEG}" 4096 4096 $C; done
  for C in 1 2 4 8 16 32; do bench $LLOG $BLOG "tier 8K/8K c$C ${LEG}" 8192 8192 $C; done
  for C in 1 2 4 8 16; do bench $LLOG $BLOG "tier 32K/32K c$C ${LEG}" 32768 32768 $C; done
  for C in 1 2 4 8; do bench $LLOG $BLOG "tier 64K/64K c$C ${LEG}" 65536 65536 $C; done

  # spec window metrics (k5 leg)
  if [ "$LEG" = "k5" ]; then
    grep -E 'SpecDecoding metrics' $BLOG | tail -300 > $OPS/spec_k5.log
  fi

  kill_all
  echo "########## GLM_MATRIX ${TAGTXT} DONE $(date -u) ##########" >> $LLOG
  echo DONE > $OPS/${LEG}_done
}

mkdir -p $OPS
echo "########## GLM MATRIX MASTER START $(date -u) ##########"
for LEG in ${LEGLIST:-k5 nospec}; do boot_leg $LEG; done
kill_all
echo "########## GLM MATRIX ALL DONE $(date -u) ##########"
echo DONE > $OPS/glm_all_done
