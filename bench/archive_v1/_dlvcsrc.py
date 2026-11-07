import os, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

out = r"E:\Develop\Deepseek\repositories\vllm-csrc"
files = open(os.path.join(out, "_filelist.txt")).read().split("\n")
files = [f for f in files if f]
BASE = "https://raw.githubusercontent.com/vllm-project/vllm/1c87991/"

def fetch(path):
    dst = os.path.join(out, path)
    if os.path.exists(dst):
        return path, "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        req = urllib.request.Request(BASE + path, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        open(dst, "wb").write(data)
        return path, len(data)
    except Exception as e:
        return path, f"ERR {type(e).__name__}"

t0 = time.time()
ok = err = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    for p, r in ex.map(fetch, files):
        if isinstance(r, str) and r.startswith("ERR"):
            err += 1
        else:
            ok += 1
print(f"done in {time.time()-t0:.1f}s: {ok} ok, {err} errors")
