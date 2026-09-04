#!/usr/bin/env python3
# 真实文本单流速度测试 (非流式, completion_tokens/总时长, 长 outputs 下 TTFT 误差<5%)
import json, time, urllib.request

URL = 'http://10.0.10.44:8123/v1/chat/completions'
HDR = {'Authorization': 'Bearer abc.12345', 'Content-Type': 'application/json'}
PROMPTS = [
    ("中文科普", "请写一篇关于现代GPU服务器散热技术演进的科普文章，从风冷讲到液冷，不少于1500字。"),
    ("代码生成", "Write a complete Python implementation of a simple HTTP load testing tool with async concurrency, statistics and percentile reporting. Full code with comments."),
    ("英文论述", "Write a detailed essay discussing the trade-offs between tensor parallelism and pipeline parallelism for large language model inference, covering communication costs, latency and throughput. At least 1200 words."),
]
for name, p in PROMPTS:
    body = json.dumps({"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": p}], "max_tokens": 2048}).encode()
    req = urllib.request.Request(URL, data=body, headers=HDR)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dur = time.time() - t0
    ct = d['usage']['completion_tokens']; pt = d['usage']['prompt_tokens']
    print(f"{name} | prompt:{pt} out:{ct} | dur:{dur:.1f}s | decode~{ct/dur:.1f} tok/s | finish:{d['choices'][0].get('finish_reason')}")
