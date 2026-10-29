import asyncio, aiohttp, json, time, sys

BASE = "http://127.0.0.1:8123/v1/chat/completions"
KEY = "abc.12345"
MODEL = "DeepSeek-V4-Pro"
# 32K token 输入
PROMPT = ("The quick brown fox jumps over the lazy dog near the riverbank "
          "while the sun sets behind the mountains, casting long shadows "
          "across the valley where a gentle stream flows steadily through "
          "the meadow, and birds sing in the tall oak trees beside the "
          "ancient stone bridge that has stood for centuries. " * 255)

async def one_req(session, idx):
    # 数数任务: 模型会持续输出数字, 不会主动 stop
    payload = {"model": MODEL,
               "messages": [{"role":"user","content": PROMPT + "\n\nCount from 1 upwards, one number per line. Do not stop until you reach the limit."}],
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
                if ttft is None and chunk.get("choices") and chunk["choices"][0].get("delta",{}).get("content"):
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
    decode_time = sum(l - t for l, t in zip(lats, ttfts)) / len(ok)
    decode_tps = out_toks / wall
    print(f"  [C={n:>3}] ok={len(ok)}/{n} fail={len(fail)} | 墙钟{wall:.1f}s | "
          f"TTFT avg{ttft_avg:.2f}s | "
          f"总延迟 avg{avg:.1f}s/max{max(lats):.1f}s | "
          f"入{in_toks}tok 出{out_toks}tok | decode吞吐{decode_tps:.1f}tok/s")

async def main():
    print(f"模型:{MODEL} | 输入~32K | 输出强制~32K(数数) | 含TTFT\n")
    for c in [int(x) for x in sys.argv[1:]]:
        await run(c); print()
    print("=== 完成 ===")
asyncio.run(main())
