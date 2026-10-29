# -*- coding: utf-8 -*-
"""Agent tool-call loop simulator for DeepSeek-V4.1-Flash (12 rounds).
Each round: user asks -> model must emit a tool call (get_weather) -> we execute
a canned result -> feed back -> model answers. Collects per-round TTFT, decode
tok/s, prompt/completion tokens; server-side DSpark acceptance read from log."""
import json, time, urllib.request

BASE = 'http://127.0.0.1:8123/v1'
KEY = '<API_KEY>'
MODEL = 'deepseek-v4.1-flash'
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}]

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆", "天津", "苏州"]


def chat(messages, tools=None, max_tokens=1024):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(BASE + '/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY,
                                          'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    return d, time.time() - t0


def main():
    rows = []
    history = [{"role": "system",
                "content": "You are a helpful assistant. Use the get_weather tool for any weather question. After receiving the tool result, answer briefly."}]
    for i, city in enumerate(CITIES):
        history.append({"role": "user",
                        "content": f"{city}现在天气怎么样？请调用工具查询，然后告诉我温度和天气状况。"})
        # round 1: expect tool call
        d, dt = chat(history, tools=TOOLS)
        m = d["choices"][0]["message"]
        u = d["usage"]
        t_round = dt
        tool_ok = bool(m.get("tool_calls"))
        ans = (m.get("content") or "").strip()
        if tool_ok:
            tc = m["tool_calls"][0]["function"]
            try:
                city_arg = json.loads(tc["arguments"]).get("city", "?")
            except Exception:
                city_arg = "?"
            history.append({"role": "assistant", "content": None,
                            "tool_calls": [{"id": m["tool_calls"][0]["id"],
                                            "type": "function", "function": tc}]})
            history.append({"role": "tool", "tool_call_id": m["tool_calls"][0]["id"],
                            "name": "get_weather",
                            "content": json.dumps({"city": city_arg, "temperature": 18 + i,
                                                   "condition": "多云转晴", "humidity": 42 + i % 20,
                                                   "wind": "东南风3级"}, ensure_ascii=False)})
            d2, dt2 = chat(history)
            m2 = d2["choices"][0]["message"]
            u2 = d2["usage"]
            ans = (m2.get("content") or "").strip()
            rows.append({"round": i + 1, "city": city, "tool_call": True, "city_arg": city_arg,
                         "t_round1_s": round(t_round, 2), "t_round2_s": round(dt2, 2),
                         "prompt_tok": u["prompt_tokens"] + u2["prompt_tokens"],
                         "completion_tok": u["completion_tokens"] + u2["completion_tokens"],
                         "ans_head": ans[:60]})
        else:
            history.append({"role": "assistant", "content": ans[:200] or "(empty)"})
            rows.append({"round": i + 1, "city": city, "tool_call": False,
                         "t_round1_s": round(t_round, 2), "t_round2_s": None,
                         "prompt_tok": u["prompt_tokens"], "completion_tok": u["completion_tokens"],
                         "ans_head": ans[:60]})
        print(rows[-1])
    n_ok = sum(1 for r in rows if r["tool_call"])
    tot_p = sum(r["prompt_tok"] for r in rows)
    tot_c = sum(r["completion_tok"] for r in rows)
    summary = {"rounds": len(rows), "tool_call_ok": n_ok,
               "total_prompt_tok": tot_p, "total_completion_tok": tot_c,
               "mean_t1_s": round(sum(r["t_round1_s"] for r in rows) / len(rows), 2),
               "mean_t2_s": round(sum(r["t_round2_s"] or 0 for r in rows if r["t_round2_s"]) / max(1, n_ok), 2),
               "rows": rows}
    with open('/root/bench_vl_dspark_1m/agent_sim.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
