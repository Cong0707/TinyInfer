import sys, os, time, glob, threading
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
files = sorted(glob.glob(os.path.join(d, '*.safetensors')))[:4]
SZ = 18_800_000
buf = bytearray(SZ)
mv = memoryview(buf)
def run(nthreads, nreads, mode):
    lock = threading.Lock()
    done = [0]
    def work(idx):
        k = 0
        while True:
            with lock:
                if done[0] >= nreads: return
                done[0] += 1; my = done[0]
            f = files[(idx + my) % len(files)]
            sz = os.path.getsize(f)
            off = (my * SZ) % max(1, sz - SZ - 1)
            if mode == 'buffered':
                with open(f, 'rb', buffering=SZ) as fh:
                    fh.seek(off); fh.readinto(mv)
            else:
                fd = os.open(f, os.O_RDONLY | getattr(os,'O_BINARY',0))
                try:
                    os.preadv(fd, [mv], off)
                finally: os.close(fd)
            k += 1
    ths = [threading.Thread(target=work, args=(i,)) for i in range(nthreads)]
    t0 = time.perf_counter()
    [t.start() for t in ths]; [t.join() for t in ths]
    dt = time.perf_counter()-t0
    return nreads*SZ/dt/1e9
for mode in ('buffered','preadv'):
    for nt in (4, 6, 8, 12, 16):
        gb = run(nt, 120, mode)
        print(f'RESULT {mode:9s} threads={nt:2d}: {gb:.2f} GB/s')