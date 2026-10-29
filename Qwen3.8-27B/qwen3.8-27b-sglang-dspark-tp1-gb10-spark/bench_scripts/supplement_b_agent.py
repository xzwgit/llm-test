#!/usr/bin/env python3
# 补充B: AI Agent 任务模拟 x12 轮 (system=工具定义, 每轮: 模型工具调用 -> 追加tool结果)
import json, time, urllib.request

URL = 'http://10.0.10.44:8123/v1/chat/completions'
HDR = {'Authorization': 'Bearer abc.12345', 'Content-Type': 'application/json'}

def stream_once(messages, max_tokens):
    body = json.dumps({"model": "Qwen3.8-27B", "messages": messages,
                       "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, data=body, headers=HDR)
    t0 = time.time(); ttft = None; usage = None; buf = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b'data: '): continue
            payload = line[6:]
            if payload == b'[DONE]': break
            try: d = json.loads(payload)
            except Exception: continue
            if d.get('usage'): usage = d['usage']
            ch = d.get('choices')
            if ch:
                delta = ch[0].get('delta', {})
                piece = delta.get('content')
                if piece: buf.append(piece)
                if ttft is None and (piece or delta.get('reasoning_content')):
                    ttft = time.time() - t0
    return ttft, time.time() - t0, usage, ''.join(buf)

SYSTEM = """You are a GPU cluster operations agent. You have these tools:

[tool: gpu_status] params: {host: string} -> returns GPU utilization, memory, temperature, power for a host.
[tool: list_models] params: {endpoint: string} -> returns deployed models with served name, quant, context length.
[tool: bench_submit] params: {model: string, in_len: int, out_len: int, conc: int} -> submits a benchmark job, returns job id.
[tool: bench_result] params: {job_id: string} -> returns throughput, TTFT/TPOT percentiles, success count.
[tool: log_tail] params: {host: string, service: string, lines: int} -> returns last N log lines.
[tool: restart_service] params: {host: string, service: string} -> restarts a service, requires confirmation.

Rules: call exactly one tool per turn as JSON like {"tool": "gpu_status", "arguments": {"host": "10.x.x.x"}}.
After collecting enough info, answer with a final summary. Be concise."""

tool_results = [
    '{"gpus": 8, "util": [92,91,93,90,88,94,92,91], "mem_used_gb": [88.2,87.9,89.1,86.4,85.2,90.3,88.8,87.6], "temp_c": [71,69,72,68,66,73,70,69], "power_w": [402,398,411,395,388,415,401,397]}',
    '{"models": [{"served": "Qwen3.8-27B", "quant": "NVFP4", "ctx": 262144, "tp": 1}, {"served": "deepseek-v4-flash", "quant": "w8a8", "ctx": 1048576, "tp": 2}]}',
    '{"job_id": "bench-20260904-001", "status": "submitted", "queue_pos": 1}',
    '{"throughput_out_tok_s": 69.1, "ttft_mean_s": 4.08, "tpot_mean_ms": 42.1, "success": 4, "total": 4}',
    '{"lines": ["[02:11:33] Decode batch #running-req: 4 accept len: 2.39", "[02:11:38] Decode batch #running-req: 4 accept len: 2.16", "[02:11:43] Decode batch #running-req: 4 accept len: 2.30"]}',
    '{"result": "service restarted, healthy in 42s"}',
]

print("=== B: Agent 任务模拟 12 轮 (system~1.5K tok, 每轮追加 tool 结果, max_tokens=200) ===")
history = [{"role": "system", "content": SYSTEM}]
task = "巡检整个推理集群: 先查 10.0.10.44 的 GPU 状态, 再列出模型, 然后提交 8K/8K c4 压测, 取结果, 看日志, 如有异常重启服务, 最后总结。"
history.append({"role": "user", "content": task})
agg_ttft, agg_dec = [], []
for i in range(12):
    ttft, dur, usage, out = stream_once(history, 200)
    pt, ct = usage['prompt_tokens'], usage['completion_tokens']
    dec = ct / (dur - ttft) if dur > ttft else 0
    agg_ttft.append(ttft); 
    if ct >= 50: agg_dec.append(dec)
    print(f"round{i+1:02d}: prompt={pt:6d} out={ct:3d} ttft={ttft:.2f}s decode={dec:5.1f} tok/s dur={dur:.1f}s | out_head: {out[:40].replace(chr(10),' ')}")
    # 追加这轮的 assistant 输出 + 对应工具结果 (循环取用)
    history.append({"role": "assistant", "content": out if out else '(empty)'})
    history.append({"role": "user", "content": "tool result: " + tool_results[i % len(tool_results)] + " continue next step."})
print(f"--- summary: ttft mean={sum(agg_ttft)/len(agg_ttft):.2f}s max={max(agg_ttft):.2f}s | decode(valid rounds) mean={sum(agg_dec)/len(agg_dec):.1f} tok/s ---")
