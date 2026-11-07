import json, urllib.request
def ask(payload):
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            o = json.loads(p)
            for c in o.get("choices") or []:
                piece = (c.get("delta") or {}).get("content") or ""
                if piece:
                    out.append(piece)
    return "".join(out)

base = {"model": "Qwen3.5-0.8B", "max_tokens": 128, "stream": True,
        "messages": [{"role": "user", "content": "从 1 到 20 里随便挑五个不重复的数，只输出数字"}]}
for name, extra in (
    ("official (t=1,k=20,pp=2.0)", {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 2.0}),
    ("greedy", {"temperature": 0.0}),
):
    txt = ask({**base, **extra})
    print(f"{name:28s} -> {txt[:90]!r}", flush=True)
