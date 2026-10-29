#!/usr/bin/env python3
"""
终极压测脚本 - 支持:
  1. 自定义输入/输出长度
  2. Input/Output/Total throughput 拆分
  3. vLLM metrics 差值: prefill/decode/queue 时间 + MTP接受率 + prefix cache命中
  4. per-request latency 分位数
用法:
  python bench_ultimate.py --in-tokens 16K --out-tokens 16K --concurrency 1 2 4 8 16
  python bench_ultimate.py --in-len 32 --out-len 32 -c 1 4    # 用 *N 倍数
"""
import asyncio, aiohttp, json, time, sys, argparse, re, urllib.request

BASE_URL = "http://127.0.0.1:8123"
COMP_URL = BASE_URL + "/v1/completions"
METRICS_URL = BASE_URL + "/metrics"
KEY = "abc.12345"
MODEL = "DeepSeek-V4-Pro"
BASE_PARA = ("The quick brown fox jumps over the lazy dog near the riverbank "
             "while the sun sets behind the mountains, casting long shadows "
             "across the valley where a gentle stream flows steadily through "
             "the meadow, and birds sing in the tall oak trees beside the "
             "ancient stone bridge that has stood for centuries. ")

def build_prompt(in_mult):
    return BASE_PARA * in_mult + "\n\nRepeat the above text verbatim, do not stop."

def fetch_metrics():
    """抓取 vLLM metrics, 返回关键计数器 dict"""
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=10) as r:
            text = r.read().decode()
    except:
        return {}
    def grab(pattern):
        m = re.findall(pattern, text)
        return sum(float(x) for x in m)
    return {
        "drafts": grab(r'vllm:spec_decode_num_drafts_total\{[^}]*\}\s+(\d+)'),
        "draft_tokens": grab(r'vllm:spec_decode_num_draft_tokens_total\{[^}]*\}\s+(\d+)'),
        "accepted": grab(r'vllm:spec_decode_num_accepted_tokens_total\{[^}]*\}\s+(\d+)'),
        "pc_queries": grab(r'vllm:prefix_cache_queries_total\{[^}]*\}\s+([\d.]+)'),
        "pc_hits": grab(r'vllm:prefix_cache_hits_total\{[^}]*\}\s+([\d.]+)'),
        "gen_tokens": grab(r'vllm:generation_tokens\{[^}]*\}\s+([\d.]+)'),
    }

async def one_req(session, prompt, max_tokens):
    payload = {"model": MODEL, "prompt": prompt,
               "max_tokens": max_tokens, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = None; out_toks = 0; in_toks = 0
    try:
        async with session.post(COMP_URL, headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"},
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
        if ttft is None: return (False, dt, 0, 0, 0, "no tokens")
        return (True, dt, ttft, in_toks, out_toks, "")
    except Exception as e:
        return (False, time.perf_counter()-t0, 0, 0, 0, str(e)[:120])

def pct(vals, p):
    if not vals: return 0
    s = sorted(vals); idx = int(len(s) * p / 100)
    return s[min(idx, len(s)-1)]

async def run(n, prompt, max_tokens):
    m_before = fetch_metrics()
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as sess:
        res = await asyncio.gather(*[one_req(sess, prompt, max_tokens) for _ in range(n)])
    wall = time.perf_counter() - t0
    m_after = fetch_metrics()
    ok = [r for r in res if r[0]]; fail = [r for r in res if not r[0]]
    if not ok:
        print(f"  [C={n}] 全部失败"); return
    lats = [r[1] for r in ok]; ttfts = [r[2] for r in ok]
    in_list = [r[3] for r in ok]; out_list = [r[4] for r in ok]
    in_total = sum(in_list); out_total = sum(out_list)
    avg = sum(lats)/len(lats)
    # metrics 差值
    dm = {k: m_after.get(k,0) - m_before.get(k,0) for k in m_before}
    mtp_rate = (dm["accepted"]/dm["draft_tokens"]*100) if dm.get("draft_tokens") else 0
    pc_rate = (dm["pc_hits"]/dm["pc_queries"]*100) if dm.get("pc_queries") else 0

    print(f"  ┌─ C={n} {'─'*50}")
    print(f"  │ Successful requests:     {len(ok)}/{n}  (failed: {len(fail)})")
    print(f"  │ Wall time:               {wall:.1f}s")
    print(f"  │")
    print(f"  │ Latency (s):   avg={avg:.2f}  p50={pct(lats,50):.2f}  p90={pct(lats,90):.2f}  p99={pct(lats,99):.2f}  max={max(lats):.2f}")
    print(f"  │ TTFT (s):      avg={sum(ttfts)/len(ttfts):.3f}  p50={pct(ttfts,50):.3f}  p90={pct(ttfts,90):.3f}  max={max(ttfts):.3f}")
    decode_time = sum(max(0, l-x) for l,x in zip(lats,ttfts)) / len(ok)
    itl_ms = (decode_time / (out_total/len(ok)) * 1000) if out_total else 0
    print(f"  │ Inter-tok lat: {itl_ms:.1f}ms/token  (decode phase avg {decode_time:.1f}s)")
    print(f"  │")
    print(f"  │ Input tokens:    total={in_total}  per-req={in_total/len(ok):.0f}")
    print(f"  │ Output tokens:   total={out_total}  per-req={out_total/len(ok):.0f}")
    print(f"  │")
    print(f"  │ Input  throughput:  {in_total/wall:.1f} tok/s")
    print(f"  │ Output throughput:  {out_total/wall:.1f} tok/s   ← decode 性能")
    print(f"  │ Total  throughput:  {(in_total+out_total)/wall:.1f} tok/s")
    print(f"  │")
    print(f"  │ MTP accept rate:    {mtp_rate:.1f}%  (draft {int(dm.get('draft_tokens',0))} → accepted {int(dm.get('accepted',0))})")
    print(f"  │ Prefix cache hit:   {pc_rate:.1f}%  (queries {int(dm.get('pc_queries',0))} → hits {int(dm.get('pc_hits',0))})")
    print(f"  └{'─'*60}")

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-len', type=int, default=125, help='输入段落倍数 (125≈16K, 255≈32K)')
    ap.add_argument('--out-tokens', type=int, default=16384, help='输出 max_tokens (16384=16K)')
    ap.add_argument('-c', '--concurrency', type=int, nargs='+', required=True, help='并发列表')
    args = ap.parse_args()
    prompt = build_prompt(args.in_len)
    in_label = {12:'~4K', 125:'~16K', 255:'~32K'}.get(args.in_len, f'x{args.in_len}')
    print(f"模型: {MODEL}")
    print(f"输入: {in_label} (段落×{args.in_len}) | 输出: max_tokens={args.out_tokens}\n")
    for c in args.concurrency:
        await run(c, prompt, args.out_tokens); print()

asyncio.run(main())
