import sys, json
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from fastapi.testclient import TestClient
from tinyinfer.server import create_app, TinyModelBackend

backend = TinyModelBackend(r"E:\Develop\Deepseek\tinyinfer\cache\weights", layers=[0, 1, 2], tier="conservative")
app = create_app(backend)
c = TestClient(app)

r = c.get("/health"); print("health", r.status_code, r.json())
r = c.get("/v1/models"); print("models", r.status_code, [m["id"] for m in r.json()["data"]])
body = {"model": "tinyinfer", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 4, "temperature": 0}
r = c.post("/v1/chat/completions", json=body)
print("chat", r.status_code)
print(json.dumps(r.json(), ensure_ascii=False)[:600])
