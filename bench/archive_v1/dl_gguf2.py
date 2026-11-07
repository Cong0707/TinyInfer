import subprocess, os, sys, time, math
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
    out = subprocess.run(["curl.exe", "-sIL", "--max-time", "60", BASE],
                         capture_output=True, text=True).stdout
    lens = []
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            try: lens.append(int(line.split(":")[1].strip()))
            except: pass
    return lens[-1]

def dl_seg(i, total):
    lo = i * SEG
    hi = min(lo + SEG, total) - 1
    want = hi - lo + 1
    path = os.path.join(DIR, f"seg_{i:04d}.part")
    if os.path.exists(path) and os.path.getsize(path) == want:
        return i, "skip"
    for attempt in range(1, 9):
        if os.path.exists(path):
            os.remove(path)
        try:
            subprocess.run(["curl.exe", "-sL", "--fail", "--retry", "5", "--retry-delay", "5",
                            "--max-time", "10800", "-r", f"{lo}-{hi}", "-o", path, BASE], timeout=11000)
        except Exception as e:
            log(f"SEG {i} attempt {attempt} exception: {e!r}")
        if os.path.exists(path) and os.path.getsize(path) == want:
            return i, "done"
        log(f"SEG {i} attempt {attempt} size={os.path.getsize(path) if os.path.exists(path) else 0}/{want}")
        time.sleep(10)
    return i, "fail"

def main():
    os.makedirs(DIR, exist_ok=True)
    total = get_total()
    nseg = math.ceil(total / SEG)
    log(f"START total={total} nseg={nseg} threads={THREADS}")
    done = 0; fail = []
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = [ex.submit(dl_seg, i, total) for i in range(nseg)]
        for f in as_completed(futs):
            i, st = f.result()
            done += 1
            if st == "fail":
                fail.append(i)
            log(f"PROGRESS {done}/{nseg} seg={i} {st}")
    if fail:
        log(f"FAILED: {fail}")
        return 2
    log("ALL SEGMENTS DONE, merging...")
    out = os.path.join(DIR, "DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf")
    if not os.path.exists(out) or os.path.getsize(out) != total:
        with open(out, "wb") as fo:
            for i in range(nseg):
                sp = os.path.join(DIR, f"seg_{i:04d}.part")
                with open(sp, "rb") as fi:
                    while True:
                        buf = fi.read(64 * 1024 * 1024)
                        if not buf: break
                        fo.write(buf)
    size = os.path.getsize(out)
    log(f"MERGE DONE size={size} match={size == total}")
    return 0 if size == total else 3

if __name__ == "__main__":
    sys.exit(main())
