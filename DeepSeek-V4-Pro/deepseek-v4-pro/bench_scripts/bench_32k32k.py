import asyncio, aiohttp, json, time, sys

BASE = "http://127.0.0.1:8123/v1/completions"
KEY = "abc.12345"
MODEL = "DeepSeek-V4-Pro"
# ~24K token 输入 (108000字符 ÷ 4.5 ≈ 24K, 实际 tokenizer 测算)
PROMPT = "The quick brown fox jumps over the lazy dog. " * 2400

async def one_req(session, idx):
    payload = {"model": MODEL,
               "prompt": PROMPT + "\n\nRepeat the above text verbatim, do not stop.",
               "max_tokens": 32768, "temperature": 0, "stream": True,
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
        print(f"  [C={n:>3}] 全部失败"); [print(f"    {r}") for r in fail[:2]]; return
    lats = [r[2] for r in ok]; ttfts = [r[3] for r in ok]
    out_toks = sum(r[5] for r in ok); in_toks = ok[0][4] * n
    avg = sum(lats)/len(lats); p50 = sorted(lats)[len(lats)//2]
    ttft_avg = sum(ttfts)/len(ttfts); ttft_p50 = sorted(ttfts)[len(ttfts)//2]
    print(f"  [C={n:>3}] ok={len(ok)}/{n} fail={len(fail)} | 墙钟{wall:.1f}s | "
          f"TTFT avg{ttft_avg:.2f}s | "
          f"总延迟 avg{avg:.1f}s/p50{p50:.1f}s/max{max(lats):.1f}s | "
          f"入{in_toks}tok 出{out_toks}tok | 总吞吐{out_toks/wall:.1f}tok/s")

async def main():
    print(f"模型:{MODEL} | 输入~24K | 输出~32K(强制) | 含TTFT\n")
    for c in [int(x) for x in sys.argv[1:]]:
        await run(c); print()
    print("=== 完成 ===")
asyncio.run(main())
