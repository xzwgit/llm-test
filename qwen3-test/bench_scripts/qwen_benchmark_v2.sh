#!/bin/bash
# Qwen3.6 压测脚本 v2 (改进: 固定样本量 + 清理间隔 + TTFT提取 + warmup)
# 用法: bash qwen_benchmark_v2.sh --p 1-8 --port 8123 [--in-len 5120 --out-len 25600]

usage() {
    echo "用法: $0 --p <起始-结束> [选项]"
    echo "  --p <a-b>       并发范围 (必填, 如 1-8)"
    echo "  --port <N>      端口 (默认 8123)"
    echo "  --in-len <N>    输入 token (默认 5120)"
    echo "  --out-len <N>   输出 token (默认 25600)"
    echo "  --model <name>  模型名 (默认 Qwen3.6-27B)"
    echo "  --prompts <N>   每档请求数 (默认 30, 越大越稳)"
    echo "  --warmup        首轮作为预热不计数 (推荐加)"
    exit 1
}

PORT=8123; RANGE=""; IN=5120; OUT=25600; MODEL="Qwen3.6-27B"; NUM_PROMPTS=30; WARMUP=0
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --p) RANGE="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --in-len) IN="$2"; shift ;;
        --out-len) OUT="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --prompts) NUM_PROMPTS="$2"; shift ;;
        --warmup) WARMUP=1 ;;
        -h|--help) usage ;;
        *) echo "❌ 未知: $1"; usage ;;
    esac
    shift
done
[ -z "$RANGE" ] && { echo "❌ 必须指定 --p"; usage; }
START=$(echo "$RANGE"|cut -d- -f1); END=$(echo "$RANGE"|cut -d- -f2)

TS=$(date +%Y%m%d_%H%M%S); DIR="./qwen_results_${TS}"; mkdir -p "$DIR"
echo "📂 $DIR | 端口 $PORT | 并发 $START-$END | 入${IN}/出${OUT} | 每档${NUM_PROMPTS}请求"
echo "============================================================"

source ~/vllm/bin/activate
SUMMARY="$DIR/summary.csv"; echo "concurrency,ttft_avg,tput_total,latency_avg,requests" > "$SUMMARY"
FIRST=1

for ((i=START; i<=END; i++)); do
    echo ""; echo "[$i/$END] 并发 $i"
    if [[ $WARMUP -eq 1 && $FIRST -eq 1 ]]; then
        echo "  (预热轮, 不计入结果)"
        WARMUP_RUN=1
    else
        WARMUP_RUN=0
    fi
    FIRST=0

    HF_ENDPOINT=https://hf-mirror.com \
    vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
        --host localhost --port "$PORT" --model "$MODEL" --tokenizer "Qwen/Qwen3.6-27B-FP8" \
        --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" \
        --request-rate inf --max-concurrency "$i" --num-prompts "$NUM_PROMPTS" \
        2>&1 | tee "${DIR}/c${i}.txt"

    # 提取关键指标到汇总
    TTFT=$(grep -oP 'TTFT.*?mean:\s*\K[\d.]+' "${DIR}/c${i}.txt" | head -1)
    TPUT=$(grep -oP 'Output token throughput.*?:\s*\K[\d.]+' "${DIR}/c${i}.txt" | head -1)
    LAT=$(grep -oP 'Mean e2e latency.*?:\s*\K[\d.]+' "${DIR}/c${i}.txt" | head -1)
    [[ $WARMUP_RUN -eq 0 ]] && echo "$i,${TTFT:-NA},${TPUT:-NA},${LAT:-NA},${NUM_PROMPTS}" >> "$SUMMARY"

    [ "$i" -lt "$END" ] && { echo "⏳ 等 15s 清理 KV cache..."; sleep 15; }
done

echo ""; echo "============================================================"
echo "📊 汇总: $SUMMARY"; cat "$SUMMARY"
