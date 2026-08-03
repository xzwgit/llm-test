import asyncio, aiohttp, json, time, sys

BASE = "http://127.0.0.1:8123/v1/chat/completions"
KEY = "abc.12345"
MODEL = "DeepSeek-V4-Pro"
PROMPT = ("The quick brown fox jumps over the lazy dog near the riverbank "
          "while the sun sets behind the mountains, casting long shadows "
          "across the valley where a gentle stream flows steadily. " * 12)

async def one_req(session, idx):
    payload = {"model": MODEL,
               "messages": [{"role":"user","content": PROMPT + "\n\nSummarize the above in detail, write about 1000 words."}],
               "max_tokens": 1024, "temperature": 0}
    t0 = time.perf_counter()
    try:
        async with session.post(BASE, headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"},
                                json=payload, timeout=aiohttp.ClientTimeout(total=900)) as r:
            data = await r.json()
            dt = time.perf_counter() - t0
            if "choices" in data:
                u = data["usage"]
                return (idx, True, dt, u["prompt_tokens"], u["completion_tokens"])
            return (idx, False, dt, 0, str(data)[:150])
    except Exception as e:
        return (idx, False, time.perf_counter()-t0, 0, str(e)[:150])

async def run(n):
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as sess:
        res = await asyncio.gather(*[one_req(sess,i) for i in range(n)])
    wall = time.perf_counter() - t0
    ok = [r for r in res if r[1]]; fail = [r for r in res if not r[1]]
    if not ok:
        print(f"  [C={n:>3}] 全部失败"); [print(f"    {r}") for r in fail[:2]]; return
    lats = [r[2] for r in ok]
    out_toks = sum(r[4] for r in ok)
    in_toks = ok[0][3] * n
    avg = sum(lats)/len(lats); p50 = sorted(lats)[len(lats)//2]
    print(f"  [C={n:>3}] ok={len(ok)}/{n} fail={len(fail)} | 墙钟{wall:.1f}s | "
          f"延迟 avg{avg:.1f}s/p50{p50:.1f}s/max{max(lats):.1f}s | "
          f"入{in_toks}tok 出{out_toks}tok | 总吞吐{out_toks/wall:.1f}tok/s | 单req{out_toks/wall/n:.1f}tok/s")

async def main():
    print(f"模型:{MODEL} | 输入~1K | 输出上限1024\n")
    for c in [int(x) for x in sys.argv[1:]]:
        await run(c); print()
    print("=== 完成 ===")
asyncio.run(main())
