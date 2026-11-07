import os, time, threading, random, glob
files = sorted(glob.glob(r"E:\models\DeepSeek-V4.1-Flash-1017\*.safetensors"))
random.seed(0)
BLOB = 18874368
def work(f, off, buf):
    fd = os.open(f, os.O_RDONLY | os.O_BINARY)
    try:
        os.lseek(fd, off, 0)
        got = 0
        while got < len(buf):
            k = os.read(fd, len(buf) - got)
            if not k: break
            buf[got:got+len(k)] = k
            got += len(k)
        return got
    finally:
        os.close(fd)

for nthreads in (4, 8, 16, 32):
    jobs = [(random.choice(files), random.randrange(0, 400_000_000)) for _ in range(96)]
    bufs = [bytearray(BLOB) for _ in range(nthreads)]
    out = [0]*len(jobs)
    idx = [0]; lock = threading.Lock()
    def worker(k):
        while True:
            with lock:
                i = idx[0]; idx[0] += 1
            if i >= len(jobs): return
            f, off = jobs[i]
            out[i] = work(f, off, bufs[k])
    ths = [threading.Thread(target=worker, args=(k,)) for k in range(nthreads)]
    t0 = time.perf_counter()
    for th in ths: th.start()
    for th in ths: th.join()
    dt = time.perf_counter() - t0
    total = sum(out)/1e9
    print(f"threads={nthreads:2d}: {total:.2f} GB in {dt:.2f}s = {total/dt:.2f} GB/s", flush=True)
