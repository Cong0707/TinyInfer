
import json, time, urllib.request, threading

URL = "http://127.0.0.1:8123/v1/chat/completions"
res = {}
def call(tag):
    body = json.dumps({"model":"DeepSeek-V4.1-Flash-1017",
                       "messages":[{"role":"user","content":"Hi"}],
                       "max_tokens":2,"temperature":0}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type":"application/json"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r: json.loads(r.read())
    except Exception: pass
    res[tag]=time.time()-t0

# 同时发 4 个
ths=[threading.Thread(target=call,args=(i,)) for i in range(4)]
t0=time.time()
for t in ths: t.start()
time.sleep(3)
# 采样 GPU
import subprocess
for i in range(6):
    u = subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader"],
                       capture_output=True,text=True).stdout.strip()
    print(f"t+{i*3}s gpu={u}")
    time.sleep(3)
for t in ths: t.join()
print("总耗时:", round(time.time()-t0,1), "s")
for k in sorted(res): print(f"  req{k}: {res[k]:.1f}s")
print("如果 4 个请求耗时相同 -> 合批成功；如果成 1x/2x/3x/4x -> 串行")
