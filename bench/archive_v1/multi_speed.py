import os, sys, time, threading, httpx, tempfile

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
FILE = "model-00001-of-00048.safetensors"
ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "https://hf-mirror.com"
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 40 * 1024 * 1024

url = f"{ENDPOINT}/{REPO}/resolve/main/{FILE}"
headers = {"User-Agent": "dsh"}

def fetch_range(lo, hi, i, results):
    t0 = time.time()
    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as c:
            r = c.get(url, headers={**headers, "Range": f"bytes={lo}-{hi}"})
            n = len(r.content)
            results[i] = (n, time.time() - t0, r.status_code)
    except Exception as e:
        results[i] = (0, time.time() - t0, str(e))

# 单线程基准
t0 = time.time()
with httpx.Client(timeout=120.0, follow_redirects=True) as c:
    r = c.get(url, headers={**headers, "Range": f"bytes=0-{CHUNK-1}"})
dt = time.time() - t0
print(f"single: {len(r.content)/1e6:.1f}MB in {dt:.2f}s -> {len(r.content)/dt/1e6:.2f} MB/s status={r.status_code}")

# 多线程
results = [None] * THREADS
threads = []
t0 = time.time()
for i in range(THREADS):
    lo = i * CHUNK
    hi = lo + CHUNK - 1
    t = threading.Thread(target=fetch_range, args=(lo, hi, i, results))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
wall = time.time() - t0
total = sum(x[0] for x in results if x)
codes = [x[2] for x in results if x]
print(f"parallel x{THREADS}: {total/1e6:.1f}MB in {wall:.2f}s -> {total/wall/1e6:.2f} MB/s codes={codes}")
