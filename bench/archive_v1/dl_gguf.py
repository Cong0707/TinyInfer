import sys, os, time, math, httpx
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://hf-mirror.com/antirez/deepseek-v4.1-flash-gguf/resolve/main/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf"
DIR = r"E:\Develop\Deepseek\deepseek\dl"
SEG = 4 * 1024**3
THREADS = 8
LOG = os.path.join(DIR, "dl.log")

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)

def get_total():
    with httpx.Client(timeout=60.0) as c:
        r = c.head(BASE, follow_redirects=True)
        return int(r.headers["content-length"])

def get_location():
    with httpx.Client(timeout=60.0, follow_redirects=False) as c:
        r = c.get(BASE)
        return r.headers["location"]

def download_seg(i, total):
    lo = i * SEG
    hi = min(lo + SEG, total) - 1
    want = hi - lo + 1
    path = os.path.join(DIR, f"seg_{i:04d}.part")
    if os.path.exists(path) and os.path.getsize(path) == want:
        return i, "skip", want
    for attempt in range(1, 8):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= want:
            return i, "done", want
        try:
            loc = get_location()
            start = lo + have
            with httpx.stream("GET", loc, headers={"Range": f"bytes={start}-{hi}"}, timeout=180.0) as r:
                if r.status_code == 200:
                    have = 0
                    start = lo
                elif r.status_code != 206:
                    raise RuntimeError(f"http {r.status_code}")
                with open(path, "ab" if have else "wb") as f:
                    for chunk in r.iter_bytes(4 * 1024 * 1024):
                        f.write(chunk)
            if os.path.getsize(path) == want:
                return i, "done", want
            log(f"SEG {i} attempt {attempt}: size {os.path.getsize(path)}/{want}, retry")
        except Exception as e:
            log(f"SEG {i} attempt {attempt} error: {e!r}")
            time.sleep(5 + attempt * 5)
    return i, "fail", want

def main():
    os.makedirs(DIR, exist_ok=True)
    total = get_total()
    nseg = math.ceil(total / SEG)
    log(f"START total={total} nseg={nseg} threads={THREADS}")
    ok = {}; fail = []
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = [ex.submit(download_seg, i, total) for i in range(nseg)]
        for f in as_completed(futs):
            i, st, want = f.result()
            ok[i] = st
            if st == "fail":
                fail.append(i)
            done = sum(1 for v in ok.values() if v in ("done", "skip"))
            log(f"PROGRESS {done}/{nseg} seg {i} {st}")
    if fail:
        log(f"FAILED segments: {fail}")
        return 2
    log("ALL SEGMENTS DONE, merging...")
    out = os.path.join(DIR, "DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf")
    if not os.path.exists(out) or os.path.getsize(out) != total:
        with open(out, "wb") as fo:
            for i in range(nseg):
                sp = os.path.join(DIR, f"seg_{i:04d}.part")
                with open(sp, "rb") as fi:
                    while True:
                        buf = fi.read(16 * 1024 * 1024)
                        if not buf: break
                        fo.write(buf)
    size = os.path.getsize(out)
    log(f"MERGE DONE size={size} match={size == total}")
    return 0 if size == total else 3

if __name__ == "__main__":
    sys.exit(main())
