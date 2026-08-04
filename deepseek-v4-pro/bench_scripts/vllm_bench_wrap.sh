#!/bin/bash
# vllm bench 封装脚本 - 循环并发档, 自动汇总
# 用法:
#   bash vllm_bench_wrap.sh --in 4096 --out 4096 --c 1 4 16 32
#   bash vllm_bench_wrap.sh --in 16384 --out 16384 --c 1 2 4 8 16
set -euo pipefail
source ~/vllm/bin/activate

IN=4096; OUT=4096; CONC=""; PORT=8123; MODEL="DeepSeek-V4-Pro"; TOK="deepseek-ai/DeepSeek-V4-Pro"
while [[ $# -gt 0 ]]; do
    case $1 in
        --in) IN="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --c) shift; while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do CONC="$CONC $1"; shift; done ;;
        --port) PORT="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --tokenizer) TOK="$2"; shift ;;
        *) echo "未知: $1"; exit 1 ;;
    esac
    shift || true
done
[ -z "$CONC" ] && { echo "必须指定 --c 并发列表"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
DIR="/root/dsv4pro/results/${MODEL}_${IN}in_${OUT}out_${TS}"
mkdir -p "$DIR"
SUMMARY="$DIR/summary.csv"
echo "concurrency,successful,failed,duration_s,input_tok,output_tok,req_tput,output_tput,total_tput,ttft_mean_ms,ttft_p99_ms,tpot_mean_ms,itl_mean_ms,mtp_accept%" > "$SUMMARY"

echo "📂 $DIR | 入${IN}/出${OUT} | 并发:$CONC"
echo "============================================================"

for C in $CONC; do
    echo ""; echo "[$C] 测试中..."
    HF_ENDPOINT=https://hf-mirror.com vllm bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --host localhost --port $PORT \
        --header Authorization="Bearer abc.12345" \
        --model $MODEL --tokenizer $TOK \
        --dataset-name random --random-input-len $IN --random-output-len $OUT \
        --request-rate inf --max-concurrency $C --num-prompts $((C < 4 ? 8 : (C > 10 ? 30 : C*3))) \
        --ignore-eos --save-result --result-dir $DIR --label "c${C}" 2>&1 | tee "$DIR/c${C}.txt" | tail -30
    # 提取 JSON
    JSON=$(ls -t $DIR/*.json 2>/dev/null | head -1)
    if [ -f "$JSON" ]; then
        python3 -c "
import json
d=json.load(open('$JSON'))
print(f'{d[\"max_concurrency\"]},{d[\"completed\"]},{d[\"failed\"]},{d[\"duration\"]:.1f},{d[\"total_input_tokens\"]},{d[\"total_output_tokens\"]},{d[\"request_throughput\"]:.2f},{d[\"output_throughput\"]:.1f},{d[\"total_token_throughput\"]:.1f},{d[\"mean_ttft_ms\"]:.0f},{d[\"p99_ttft_ms\"]:.0f},{d[\"mean_tpot_ms\"]:.1f},{d[\"mean_itl_ms\"]:.1f},{d.get(\"spec_decode_acceptance_rate\",0):.1f}', file=open('$SUMMARY','a'))
"
    fi
    [ "$C" != "$(echo $CONC | awk '{print $NF}')" ] && { echo "⏳ 等15s..."; sleep 15; }
done

echo ""; echo "============================================================"
echo "📊 汇总: $SUMMARY"
echo "---"
cat "$SUMMARY" | column -t -s,
