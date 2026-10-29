import asyncio, aiohttp, json, time, sys

BASE = "http://127.0.0.1:8123/v1/completions"
KEY = "abc.12345"
MODEL = "DeepSeek-V4-Pro"
PROMPT = ("The quick brown fox jumps over the lazy dog near the riverbank "
          "while the sun sets behind the mountains, casting long shadows "
          "across the valley where a gentle stream flows steadily through "
          "the meadow, and birds sing in the tall oak trees beside the "
          "ancient stone bridge that has stood for centuries. " * 125)

async def one_req(session, idx):
    payload = {"model": MODEL,
               "prompt": PROMPT + "\n\nRepeat the above text verbatim, do not stop.",
               "max_tokens": 16384, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = None; out_toks = 0; in_toks = 0
    try:
        async with session.post(BASE, headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"},
                                json=payload, timeout=aiohttp.ClientTimeout(total=3600)) as r:
            async for raw in r.content:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "): continue
                data = line[6:]
                if data == "[DONE]": break
                try: chunk = json.loads(data)
                except: continue
                if ttft is None and chunk.get("choices") and chunk["choices"][0].get("text"):
                    ttft = time.perf_counter() - t0
                if chunk.get("usage"):
                    out_toks = chunk["usage"].get("completion_tokens", 0)
                    in_toks = chunk["usage"].get("prompt_tokens", 0)
        dt = time.perf_counter() - t0
        if ttft is None: return (idx, False, dt, 0, 0, 0, "no tokens")
        return (idx, True, dt, ttft, in_toks, out_toks, "")
    except Exception as e:
        return (idx, False, time.perf_counter()-t0, 0, 0, 0, str(e)[:120])

async def run(n):
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as sess:
        res = await asyncio.gather(*[one_req(sess,i) for i in range(n)])
    wall = time.perf_counter() - t0
    ok = [r for r in res if r[1]]; fail = [r for r in res if not r[1]]
    if not ok:
        print(f"  [C={n}] 全部失败"); return
    lats = [r[2] for r in ok]; ttfts = [r[3] for r in ok]
    in_list = [r[4] for r in ok]; out_list = [r[5] for r in ok]
    in_total = sum(in_list); out_total = sum(out_list)
    avg = sum(lats)/len(lats); p50 = sorted(lats)[len(lats)//2]
    ttft_avg = sum(ttfts)/len(ttfts); ttft_p50 = sorted(ttfts)[len(ttfts)//2]

    # 专业格式输出 (类似 vllm bench / SGLang 报告)
    print(f"  ┌─ C={n} {'─'*40}")
    print(f"  │ Successful requests:     {len(ok)}/{n}  (failed: {len(fail)})")
    print(f"  │")
    print(f"  │ Request latency (s):  avg={avg:.2f}  p50={p50:.2f}  max={max(lats):.2f}")
    print(f"  │ TTFT (s):              avg={ttft_avg:.3f}  p50={ttft_p50:.3f}  max={max(ttfts):.3f}")
    print(f"  │ Wall time (s):         {wall:.1f}")
    print(f"  │")
    print(f"  │ Input tokens:    total={in_total}  per-req={in_total/len(ok):.0f}")
    print(f"  │ Output tokens:   total={out_total}  per-req={out_total/len(ok):.0f}")
    print(f"  │")
    print(f"  │ Input  throughput:  {in_total/wall:.1f} tok/s")
    print(f"  │ Output throughput:  {out_total/wall:.1f} tok/s   ← decode 性能")
    print(f"  │ Total  throughput:  {(in_total+out_total)/wall:.1f} tok/s")
    print(f"  └{'─'*50}")

async def main():
    print(f"模型: {MODEL}")
    print(f"输入: ~16K (强制), 输出: ~16K (强制 max_tokens=16384)\n")
    for c in [int(x) for x in sys.argv[1:]]:
        await run(c); print()
    print("=== 完成 ===")
asyncio.run(main())
