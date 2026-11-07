import sys, time, threading, urllib.request

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
FILE = "model-00001-of-00048.safetensors"
ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "https://hf-mirror.com"
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 40 * 1024 * 1024
url = f"{ENDPOINT}/{REPO}/resolve/main/{FILE}"

def get_range(lo, hi, out=None):
    req = urllib.request.Request(url, headers={"User-Agent": "dsh", "Range": f"bytes={lo}-{hi}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if out:
        with open(out, "wb") as f:
            f.write(data)
    return len(data)

t0 = time.time()
n = get_range(0, CHUNK - 1)
dt = time.time() - t0
print(f"single: {n/1e6:.1f}MB in {dt:.2f}s -> {n/dt/1e6:.2f} MB/s")

results = [0] * THREADS
def worker(i):
    results[i] = get_range(i * CHUNK, i * CHUNK + CHUNK - 1)
threads = [threading.Thread(target=worker, args=(i,)) for i in range(THREADS)]
t0 = time.time()
for t in threads: t.start()
for t in threads: t.join()
wall = time.time() - t0
total = sum(results)
print(f"parallel x{THREADS}: {total/1e6:.1f}MB in {wall:.2f}s -> {total/wall/1e6:.2f} MB/s per_thread={[round(r/1e6,1) for r in results]}")
