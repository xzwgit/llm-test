#!/bin/bash
# Stop vLLM cleanly, verify GPUs free, run NCCL 8-GPU all_reduce baseline.
set -u
echo "== stopping vllm =="
for p in $(pgrep -f 'bin/[v]llm serve'); do kill $p; done
sleep 8
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
pgrep -f 'vllm-r[s]' | xargs -r kill -9 2>/dev/null
rm -f /dev/shm/nccl-* 2>/dev/null
sleep 3
echo "gpu procs left: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8
exit 0
