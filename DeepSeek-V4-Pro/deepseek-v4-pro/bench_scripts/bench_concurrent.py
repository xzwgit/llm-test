import asyncio, aiohttp, json, time, sys, argparse

BASE = "http://127.0.0.1:8123/v1/chat/completions"
KEY = "abc.12345"
MODEL = "DeepSeek-V4-Pro"

# ~1K token 输入: 一段重复英文, 约 1000 token
PROMPT = ('The quick brown fox jumps over the lazy dog near the riverbank '
          'while the sun sets behind the mountains, casting long shadows '
          'across the valley. ' * 10)

async def one_request(session, idx):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT + "\n\nSummarize the above in detail, output exactly 1000 words."}],
        "max_tokens": 1024,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(BASE, headers={"Authorization": f"Bearer {KEY}",
                                                "Content-Type": "application/json"},
                                json=payload, timeout=aiohttp.ClientTimeout(total=600)) as r:
            data = await r.json()
            dt = time.perf_counter() - t0
            if "choices" in data:
                ct = data["usage"]["completion_tokens"]
                pt = data["usage"]["prompt_tokens"]
                return (idx, True, dt, pt, ct)
            else:
                return (idx, False, dt, 0, str(data)[:120])
    except Exception as e:
        return (idx, False, time.perf_counter()-t0, 0, str(e)[:120])

async def run(n):
    t0 = time.perf_counter()
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as sess:
        results = await asyncio.gather(*[one_request(sess, i) for i in range(n)])
    wall = time.perf_counter() - t0
    ok = [r for r in results if r[1]]
    fail = [r for r in results if not r[1]]
    if not ok:
        print(f"  [C={n}] 全部失败"); 
        for r in fail[:3]: print(f"    {r}")
        return
    tts = [r[2] for r in ok]
    out_toks = sum(r[4] for r in ok)
    in_toks = sum(r[3] for r in ok)
    avg_lat = sum(tts)/len(tts)
    max_lat = max(tts)
    out_throughput = out_toks / wall
    per_req_tps = out_throughput / n
    print(f"  [C={n:>3}] 成功{len(ok)}/{n} 失败{len(fail)} | 墙钟{wall:.1f}s | "
          f"平均延迟{avg_lat:.1f}s 最大{max_lat:.1f}s | "
          f"输入{in_toks}tok 输出{out_toks}tok | "
          f"总输出吞吐{out_throughput:.1f} tok/s | 单请求{per_req_tps:.1f} tok/s")

async def main():
    cs = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8]
    print(f"输入tokens~{len(PROMPT.split())}词 | 输出上限1024 | 模型{MODEL}\n")
    for c in cs:
        await run(c)
        print()
    print("=== 测试完成 ===")

asyncio.run(main())
