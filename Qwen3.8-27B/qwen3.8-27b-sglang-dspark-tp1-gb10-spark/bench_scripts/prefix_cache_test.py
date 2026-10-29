#!/usr/bin/env python3
# prefix cache 命中验证: 同一 8K prompt 发 3 次, max_tokens=1, 总时长 ~= TTFT
import json, time, urllib.request

URL = 'http://10.0.10.44:8123/v1/chat/completions'
HDR = {'Authorization': 'Bearer abc.12345', 'Content-Type': 'application/json'}

base = ("GPU server thermal design has evolved from air cooling to liquid cooling. "
        "Modern accelerators demand dense heat dissipation solutions. ")
# 填到 ~8K token (英文 ~0.75 词/token, 词长~6.3字符)
prompt = base * 600
body_t = json.dumps({"model": "Qwen3.8-27B",
                     "messages": [{"role": "user", "content": prompt + " Summarize in one word."}],
                     "max_tokens": 1})
print(f"prompt chars: {len(prompt)}")

for i in range(3):
    t0 = time.time()
    req = urllib.request.Request(URL, data=body_t.encode(), headers=HDR)
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    print(f"run{i+1}: {dt:.2f}s | prompt_tokens: {d['usage']['prompt_tokens']}")
