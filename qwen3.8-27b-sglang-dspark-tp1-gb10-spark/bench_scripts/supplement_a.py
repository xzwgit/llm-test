#!/usr/bin/env python3
# 补充A1: 多轮对话模拟 (流式, TTFT/decode分离) + A2: 热态重复前缀长输出
import json, time, urllib.request

URL = 'http://10.0.10.44:8123/v1/chat/completions'
HDR = {'Authorization': 'Bearer abc.12345', 'Content-Type': 'application/json'}

def stream_once(messages, max_tokens=64):
    body = json.dumps({"model": "Qwen3.8-27B", "messages": messages,
                       "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, data=body, headers=HDR)
    t0 = time.time(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b'data: '): continue
            payload = line[6:]
            if payload == b'[DONE]': break
            try: d = json.loads(payload)
            except Exception: continue
            if d.get('usage'): usage = d['usage']
            if ttft is None:
                ch = d.get('choices')
                if ch and (ch[0].get('delta', {}).get('content') or ch[0].get('delta', {}).get('reasoning_content')):
                    ttft = time.time() - t0
    return ttft, time.time() - t0, usage

# ---- A1: 多轮对话 (8K 文档做 system, 逐轮追加) ----
doc = ("NVIDIA DGX Spark is a compact AI workstation based on the GB10 Grace Blackwell "
       "superchip, featuring 121GB of unified coherent memory. ") * 640  # ~8K token
rounds_q = [
    "这套机器的统一内存有什么优势?",
    "它适合跑什么规模的模型?",
    "和标准机架服务器比有什么取舍?",
    "总结前面对话的三个要点。",
    "再补充一条关于扩展性的建议。",
]
print("=== A1: 多轮对话 (system=8K文档, 逐轮追加, max_tokens=64) ===")
history = []
for i, q in enumerate(rounds_q):
    msgs = [{"role": "system", "content": doc}] + history + [{"role": "user", "content": q}]
    ttft, dur, usage = stream_once(msgs, 64)
    print(f"round{i+1}: prompt={usage['prompt_tokens']} ttft={ttft:.2f}s decode={usage['completion_tokens']/(dur-ttft):.1f} tok/s dur={dur:.1f}s")
    # 造一轮假回答进历史 (真实感)
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": "简答: " + q + " 的要点如上所述。" * 3})

# ---- A2: 热态重复前缀, 3 个不同问题, 各 1024 输出 ----
print("=== A2: 热态同前缀长输出 (system=8K文档, 输出~1024) ===")
warm_msgs = [{"role": "system", "content": doc}, {"role": "user", "content": "warmup"}]
stream_once(warm_msgs, 8)
qs = ["详细解释统一内存架构的技术细节", "分析这类机器的目标用户和场景", "论述它与分布式集群的互补关系"]
for q in qs:
    msgs = [{"role": "system", "content": doc}, {"role": "user", "content": q}]
    ttft, dur, usage = stream_once(msgs, 1024)
    pt, ct = usage['prompt_tokens'], usage['completion_tokens']
    print(f"'{q[:14]}...': prompt={pt} out={ct} ttft={ttft:.2f}s decode={ct/(dur-ttft):.1f} tok/s | 总吞吐={(pt+ct)/dur:.1f} tok/s | 输出吞吐={ct/dur:.1f} tok/s dur={dur:.1f}s")
