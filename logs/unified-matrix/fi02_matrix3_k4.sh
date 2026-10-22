#!/bin/bash
export PATH=/usr/local/cuda/bin:${PATH}
# fi02_matrix3_k4.sh — two phases, both at gpu-memory-utilization 0.90:
#   Phase 1: k=4 full matrix re-run (after 05:54 OOM at util 0.92/capture 768).
#            capture 640 (=128*(4+1) exact), engine health check per bench,
#            one automatic reboot+ladder-restart on engine death, 1K c1 double-run.
#   Phase 2: k=3 agent redo (v1's k3 agent sessions all failed with 404:
#            agent_tools_sim.py now URL-normalized, fixed 07:14 UTC).
#            Boot k=3 (capture 512), 3 agent sessions overwrite matrix_k3_agent_s*.out.
# All markers written at the VERY END so the downstream watcher sees one completion point:
#   matrix_k4_v3.done (k4 phase status), matrix_k3agent_v3.done (k3 agent phase status),
#   matrix_k4.done (DONE for parser-side bookkeeping).
set -u
V=/root/vllm-fi02
MODEL=/data/models/DeepSeek-V4.1-Flash-1017
OPS=/root/test_ops
export OPENAI_API_KEY=*** HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 FLASHINFER_DISABLE_VERSION_CHECK=1

kill_all () {
  pkill -9 -f "vllm serve" 2>/dev/null
  pkill -9 -f "VLLM::" 2>/dev/null
  sleep 2
  for p in $(ss -lptnH "sport = :8123" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do
    kill -9 "$p" 2>/dev/null
  done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done
  rm -f /dev/shm/nccl-* /dev/shm/vllm-*
  sleep 3
}

engine_alive () {
  curl -s -m 20 http://127.0.0.1:8123/v1/models -H "Authorization: Bearer ***" 2>/dev/null | grep -q deepseek
}

bench () {
  # bench <log> <label> <ilen> <olen> <conc>; returns 90 on engine death
  local LOG="$1" label="$2" ilen="$3" olen="$4" conc="$5"
  if ! engine_alive; then
    echo "!!!!! $label ENGINE DEAD BEFORE BENCH $(date -u +%H:%M:%S)" >> $LOG
    return 90
  fi
  echo "===== $label $(date -u +%H:%M:%S) =====" >> $LOG
  timeout 5400 env OPENAI_API_KEY=*** $V/bin/vllm bench serve \
    --base-url http://127.0.0.1:8123 \
    --model deepseek-v4.1-flash \
    --tokenizer $MODEL \
    --dataset-name random \
    --random-input-len $ilen --random-output-len $olen \
    --ignore-eos \
    --num-prompts $conc --request-rate $conc \
    >> $LOG 2>&1
  local RC=$?
  if [ $RC -ne 0 ]; then
    echo "!!!!! $label BENCH EXIT $RC (timeout/err) $(date -u +%H:%M:%S)" >> $LOG
  fi
  if ! engine_alive; then
    echo "!!!!! $label ENGINE DEAD AFTER BENCH $(date -u +%H:%M:%S)" >> $LOG
    return 90
  fi
  return 0
}

boot_k () {
  # boot_k <K> <CAP> <UTIL> <BLOG> <KLOG>; returns 0 on ready
  local K="$1" CAP="$2" UTIL="$3" BLOG="$4" KLOG="$5"
  kill_all
  echo "########## MATRIX k=$K BOOT (v3 capture=$CAP util=$UTIL) $(date -u) ##########" > $KLOG
  echo "env: $(cd /root && $V/bin/python -c 'import vllm,flashinfer,torch; print(vllm.__version__, flashinfer.__version__, torch.__version__)')" >> $KLOG
  echo "BOOT CMD: $V/bin/vllm serve $MODEL --served-model-name deepseek-v4.1-flash --tokenizer-mode deepseek_v41 --trust-remote-code --tensor-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8 --attention-config {backend:FLASHINFER_MLA_SPARSE_DSV41,indexer_kv_dtype:fp8} --engram-config {cpu_offload:true} --moe-backend auto --compilation-config {cudagraph_mode:FULL} --language-model-only --max-model-len 1048576 --max-num-seqs 128 --gpu-memory-utilization $UTIL --speculative-config {method:dspark,num_speculative_tokens:$K} --max-cudagraph-capture-size $CAP --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --port 8123 --api-key ***" >> $KLOG

  setsid nohup env VLLM_USE_RUST_FRONTEND=1 VLLM_ENGINE_READY_TIMEOUT_S=7200 \
    NCCL_P2P_LEVEL=SYS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
    $V/bin/vllm serve $MODEL \
    --served-model-name deepseek-v4.1-flash \
    --tokenizer-mode deepseek_v41 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"fp8"}' \
    --engram-config '{"cpu_offload":true}' \
    --moe-backend auto \
    --compilation-config '{"cudagraph_mode": "FULL"}' \
    --language-model-only \
    --max-model-len 1048576 \
    --max-num-seqs 128 \
    --gpu-memory-utilization $UTIL \
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$K}" \
    --max-cudagraph-capture-size $CAP \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --port 8123 --api-key *** \
    > $BLOG 2>&1 < /dev/null &

  for i in $(seq 1 100); do
    sleep 10
    ANS=$(curl -s -m 60 http://127.0.0.1:8123/v1/completions \
      -H "Authorization: Bearer ***" -H "Content-Type: application/json" \
      -d '{"model":"deepseek-v4.1-flash","prompt":"17*19=","max_tokens":8,"temperature":0}' \
      2>/dev/null | grep -o '"text":"[^"]*"' | head -1)
    if echo "$ANS" | grep -q 323; then
      echo "k=$K READY after ~$((i*10))s $(date -u +%H:%M:%S)" >> $KLOG
      grep -E 'KV cache size|DSpark draft' $BLOG | tail -3 >> $KLOG
      return 0
    fi
    if grep -qE 'Assertion error|ValueError|Engine core initialization failed|OutOfMemory|CUDA out of memory' $BLOG 2>/dev/null; then
      echo "k=$K BOOT FAILED:" >> $KLOG
      grep -m5 -E 'Assertion|ValueError|OutOfMemory|Engine core' $BLOG >> $KLOG
      return 1
    fi
  done
  echo "k=$K BOOT TIMEOUT" >> $KLOG
  return 1
}

agent_leg () {
  # agent_leg <K>; assumes engine up; returns 0 if all 3 sessions produced data
  local K="$1" OKCNT=0
  echo "########## k=$K AGENT START $(date -u) ##########" >> $OPS/matrix_k$K.log
  for S in 1 2 3; do
    $V/bin/python $OPS/agent_tools_sim.py http://127.0.0.1:8123 --turns 12 > $OPS/matrix_k${K}_agent_s$S.out 2>&1
    echo "agent s$S rc=$? $(date -u +%H:%M:%S)" >> $OPS/matrix_k$K.log
    if grep -q '"turns": 12' $OPS/matrix_k${K}_agent_s$S.out 2>/dev/null; then
      OKCNT=$((OKCNT+1))
    fi
  done
  [ $OKCNT -eq 3 ]
}

# ---------------- Phase 1: k=4 matrix re-run ----------------
echo "########## MATRIX3 phase1 k=4 START $(date -u) ##########"
K4=4
CAP4=640
UTIL4=0.90
BLOG4=$OPS/matrix_boot_k4.log
KLOG4=$OPS/matrix_k4.log
rm -f $OPS/matrix_k4.done $OPS/matrix_k4_v3.done $OPS/matrix_k3agent_v3.done

K4_STATUS="FAIL"
ATTEMPT=0
while [ $ATTEMPT -lt 2 ]; do
  ATTEMPT=$((ATTEMPT+1))
  if ! boot_k "$K4" "$CAP4" "$UTIL4" "$BLOG4" "$KLOG4"; then
    echo "!!!!! phase1 boot failed attempt $ATTEMPT" >> $KLOG4
    break
  fi
  sleep 5
  echo "===== correctness k=$K4 =====" >> $KLOG4
  curl -s -m 120 http://127.0.0.1:8123/v1/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer ***" \
    -d '{"model":"deepseek-v4.1-flash","prompt":"17*19=","max_tokens":16,"temperature":0}' >> $KLOG4 2>&1
  echo >> $KLOG4

  DEAD=0
  bench $KLOG4 "tier 1K/1K c1 k=$K4" 1024 1024 1 || DEAD=1
  bench $KLOG4 "tier 1K/1K c1 k=$K4" 1024 1024 1 || DEAD=1
  for C in 2 4 8 16 32 64 128; do [ $DEAD -eq 0 ] && bench $KLOG4 "tier 1K/1K c$C k=$K4" 1024 1024 $C || DEAD=1; done
  for C in 1 2 4 8 16 32;    do [ $DEAD -eq 0 ] && bench $KLOG4 "tier 4K/4K c$C k=$K4" 4096 4096 $C || DEAD=1; done
  for C in 1 2 4 8 16 32;    do [ $DEAD -eq 0 ] && bench $KLOG4 "tier 8K/8K c$C k=$K4" 8192 8192 $C || DEAD=1; done
  for C in 1 2 4 8 16;       do [ $DEAD -eq 0 ] && bench $KLOG4 "tier 32K/32K c$C k=$K4" 32768 32768 $C || DEAD=1; done
  for C in 1 2 4 8;          do [ $DEAD -eq 0 ] && bench $KLOG4 "tier 64K/64K c$C k=$K4" 65536 65536 $C || DEAD=1; done

  if [ $DEAD -eq 0 ] && engine_alive; then
    if agent_leg "$K4"; then
      grep -E 'SpecDecoding metrics' $BLOG4 | tail -150 > $OPS/matrix_k${K4}_spec_metrics.log
      echo "########## MATRIX3 k=$K4 DONE attempt=$ATTEMPT $(date -u) ##########" >> $KLOG4
      K4_STATUS="DONE"
      break
    else
      echo "!!!!! k=$K4 agent leg incomplete (engine dead?), attempt $ATTEMPT" >> $KLOG4
    fi
  else
    echo "!!!!! k=$K4 ladder failed (engine death), attempt $ATTEMPT $(date -u)" >> $KLOG4
  fi
done

# ---------------- Phase 2: k=3 agent redo (404 fix) ----------------
echo "########## MATRIX3 phase2 k=3 agent redo START $(date -u) ##########"
K3A_STATUS="FAIL"
if boot_k 3 512 0.90 $OPS/matrix_boot_k3agent.log $OPS/matrix_k3agent.log; then
  sleep 5
  echo "===== correctness k=3 (agent redo boot) =====" >> $OPS/matrix_k3agent.log
  curl -s -m 120 http://127.0.0.1:8123/v1/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer ***" \
    -d '{"model":"deepseek-v4.1-flash","prompt":"17*19=","max_tokens":16,"temperature":0}' >> $OPS/matrix_k3agent.log 2>&1
  echo >> $OPS/matrix_k3agent.log
  if agent_leg 3; then
    grep -E 'SpecDecoding metrics' $OPS/matrix_boot_k3agent.log | tail -150 > $OPS/matrix_k3agent_spec_metrics.log
    K3A_STATUS="DONE"
  fi
else
  echo "!!!!! phase2 k=3 boot failed" >> $OPS/matrix_k3agent.log
fi

# ---------------- final markers ----------------
kill_all
echo "########## MATRIX3 END k4=$K4_STATUS k3agent=$K3A_STATUS $(date -u) ##########"
echo "$K4_STATUS" > $OPS/matrix_k4.done
echo "$K4_STATUS" > $OPS/matrix_k4_v3.done
echo "$K3A_STATUS" > $OPS/matrix_k3agent_v3.done
