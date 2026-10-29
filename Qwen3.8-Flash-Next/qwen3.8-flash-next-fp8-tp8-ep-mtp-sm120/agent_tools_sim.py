"""Real tools-API agent simulator: sends `tools` in every request, verifies
structured tool_calls come back (this is the path a plain-text sim never hits).
Usage: python agent_tools_sim.py <base_url> [turns]"""
import argparse
import json
import sys
import time
import urllib.request

URL = None
KEY = "abc.12345"
MODEL = "qwen3.8-flash-next"

TOOLS = [
    {"type": "function", "function": {
        "name": "get_gpu_alert",
        "description": "查询指定 GPU 节点的当前告警状态",
        "parameters": {"type": "object",
                       "properties": {"node": {"type": "string",
                                               "description": "节点名，如 gpu-03"}},
                       "required": ["node"]}}},
    {"type": "function", "function": {
        "name": "query_metrics",
        "description": "查询指定节点的某项监控指标最近 10 分钟均值",
        "parameters": {"type": "object",
                       "properties": {"node": {"type": "string"},
                                      "metric": {"type": "string",
                                                 "enum": ["gpu_util", "mem_used", "temperature"]}},
                       "required": ["node", "metric"]}}},
    {"type": "function", "function": {
        "name": "scale_node",
        "description": "扩容或缩容指定节点上的服务副本数",
        "parameters": {"type": "object",
                       "properties": {"node": {"type": "string"},
                                      "replicas": {"type": "integer"}},
                       "required": ["node", "replicas"]}}},
]

def canned(name, argsd):
    node = argsd.get("node", "gpu-03")
    if name == "get_gpu_alert":
        return json.dumps({"node": node, "alerts": [
            {"level": "warn", "msg": "GPU5 util 97% 持续 8 分钟"},
            {"level": "info", "msg": "驱动版本 595.71.05"}]}, ensure_ascii=False)
    if name == "query_metrics":
        return json.dumps({"node": node,
                           "metric": argsd.get("metric", "gpu_util"),
                           "avg_10min": 73.5, "peak": 98.2}, ensure_ascii=False)
    if name == "scale_node":
        return json.dumps({"node": node,
                           "replicas": argsd.get("replicas", 2),
                           "status": "ok"}, ensure_ascii=False)
    return "{}"

FOLLOWUPS = [
    "再帮我查一下 gpu-03 的温度。",
    "把 gpu-03 的服务副本扩到 4 个。",
    "顺便查 gpu-07 的显存占用。",
    "gpu-07 有没有告警？",
    "总结一下两个节点的状况，用两句话说。",
]


def chat(messages, tmo=180):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "tools": TOOLS, "tool_choice": "auto",
                       "max_tokens": 512, "temperature": 0}).encode()
    req = urllib.request.Request(
        URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=tmo) as r:
        d = json.load(r)
    return d, round(time.time() - t0, 3)


def main() -> int:
    global URL
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--turns", type=int, default=6)
    args = ap.parse_args()
    URL = args.base_url

    messages = [{"role": "user",
                 "content": "你是运维助手，需要查询信息时必须调用工具。现在：帮我查一下 gpu-03 的告警状态。"}]
    stats = {"turns": 0, "tool_call_turns": 0, "raw_xml_leak": 0,
             "errors": 0, "latencies": [], "calls_by_tool": {}}

    for t in range(args.turns):
        try:
            d, lat = chat(messages)
        except Exception as e:
            stats["errors"] += 1
            print(json.dumps({"error_at_turn": t, "detail": str(e)[:200]},
                             ensure_ascii=False))
            break
        stats["turns"] += 1
        stats["latencies"].append(lat)
        msg = d["choices"][0]["message"]

        if msg.get("tool_calls"):
            stats["tool_call_turns"] += 1
            messages.append({"role": "assistant",
                             "content": msg.get("content") or "",
                             "tool_calls": msg["tool_calls"]})
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                stats["calls_by_tool"][name] = stats["calls_by_tool"].get(name, 0) + 1
                try:
                    argsd = json.loads(tc["function"]["arguments"])
                except Exception:
                    argsd = {}
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": canned(name, argsd)})
        else:
            content = msg.get("content") or ""
            if "<tool_call>" in content or "<function=" in content:
                stats["raw_xml_leak"] += 1
            messages.append({"role": "assistant", "content": content})
            if t < len(FOLLOWUPS):
                messages.append({"role": "user", "content": FOLLOWUPS[t]})

    ok = stats["tool_call_turns"] > 0 and stats["raw_xml_leak"] == 0 and stats["errors"] == 0
    print(json.dumps({
        **stats,
        "lat_mean_s": round(sum(stats["latencies"]) / max(1, len(stats["latencies"])), 3),
        "structured_tool_calls_ok": ok,
    }, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
