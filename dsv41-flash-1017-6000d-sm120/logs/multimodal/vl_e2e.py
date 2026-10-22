"""E2E multimodal capability bench: per image size (512/1024/2048) x concurrency (1/4),
plus one text baseline. Streaming chat to measure TTFT + total; content graded
(red circle + text token). Prints a compact table."""
import base64
import io
import json
import re
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8123/v1/chat/completions"
KEY = "***"
MODEL = "deepseek-v4.1-flash"

from PIL import Image, ImageDraw


def make_img(size):
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    r = int(size * 0.22)
    cx = size // 2
    d.ellipse([cx - r, int(size * 0.12), cx + r, int(size * 0.12) + 2 * r], fill="red")
    d.text((cx - int(size * 0.18), int(size * 0.72)), "VLM TEST 7", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def one_request(b64, ttfts, totals, oks, idx):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "描述这张图:有什么形状颜色?把图中文字原样写出来。"},
        ]}],
        "max_tokens": 160, "temperature": 0, "stream": True,
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    ttft = None
    text = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except Exception:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta", {})
            piece = delta.get("content") or ""
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                text.append(piece)
    total = time.time() - t0
    ttfts[idx] = ttft or total
    totals[idx] = total
    full = "".join(text)
    ok = ("红" in full and "圆" in full) and re.search(r"VLM\s*TEST\s*7", full) is not None
    oks[idx] = ok


def run_case(b64, conc):
    ttfts, totals, oks = [None] * conc, [None] * conc, [None] * conc
    ths = [threading.Thread(target=one_request, args=(b64, ttfts, totals, oks, i)) for i in range(conc)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    return wall, ttfts, totals, oks


print(f"{'case':<14}{'wall_s':>7}{'ttft_s':>8}{'total_s':>8}{'ok':>5}")
# text baseline (no image)
t0 = time.time()
body = {"model": MODEL, "messages": [{"role": "user", "content": "用两句话介绍你自己。"}],
        "max_tokens": 160, "temperature": 0}
req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {KEY}"})
with urllib.request.urlopen(req, timeout=300) as r:
    json.load(r)
print(f"{'text-baseline':<14}{time.time()-t0:>7.2f}{'-':>8}{time.time()-t0:>8.2f}{'-':>5}")

for size in (512, 1024, 2048):
    b64 = make_img(size)
    for conc in (1, 4):
        wall, ttfts, totals, oks = run_case(b64, conc)
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_total = sum(totals) / len(totals)
        print(f"img{size}x{size} c{conc:<3}{wall:>7.2f}{avg_ttft:>8.2f}{avg_total:>8.2f}{sum(oks):>3}/{conc}")
print("E2E_DONE")
