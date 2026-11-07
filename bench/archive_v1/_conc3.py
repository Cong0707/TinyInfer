
import json, time, urllib.request, threading, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")

URL = "http://127.0.0.1:8123/v1/chat/completions"
def call(tag):
    body = json.dumps({"model":"DeepSeek-V4.1-Flash-1017",
                       "messages":[{"role":"user","content":"Hi"}],
                       "max_tokens":1,"temperature":0}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r: json.loads(r.read())
    except Exception as e: pass

# 先全部塞进去，再一起放行：用线程同时 start，确保都在 inbox 里
ths=[threading.Thread(target=call,args=(i,)) for i in range(4)]
t0=time.time()
for t in ths: t.start()
time.sleep(2)
# 立刻查 health 的 active_requests
try:
    h = json.loads(urllib.request.urlopen("http://127.0.0.1:8123/health", timeout=5).read())
    print("active_requests:", h.get("active_requests"))
except Exception as e:
    print("health err", e)
for t in ths: t.join()
print("总耗时:", round(time.time()-t0,1))
