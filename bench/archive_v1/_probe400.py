import json, urllib.request, urllib.error
payload = {
    "model": "Qwen3.5-0.8B",
    "max_tokens": 2000,
    "temperature": 1.0, "top_p": 1.0, "top_k": 20,
    "presence_penalty": 2.0, "repetition_penalty": 1.0,
    "stream": True,
    "messages": [{"role": "user", "content": "连续输出数字，每个一行，不要停"}],
}
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print("status", r.status)
        head = r.read(400).decode("utf-8", "replace")
        print(head[:400])
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code)
    print(exc.read().decode("utf-8", "replace")[:600])
