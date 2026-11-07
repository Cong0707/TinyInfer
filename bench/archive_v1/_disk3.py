import sys, os, time, glob, threading
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
files = sorted(glob.glob(os.path.join(d, '*.safetensors')))
SZ = 18_800_000
buf = bytearray(SZ); mv = memoryview(buf)
def run(nthreads, nreads, avoid_pages):
    lock = threading.Lock(); state = {'i': 0}
    def work():
        while True:
            with lock:
                my = state['i']
                if my >= nreads: return
                state['i'] += 1
            f = files[(my * 7) % len(files)]
            sz = os.path.getsize(f)
            # avoid_pages: 每次都跳到文件不同深度，避免页缓存
            if avoid_pages:
                off = ((my * 37) % 40) * SZ if sz > 40*SZ else (my*SZ) % max(1, sz-SZ-1)
            else:
                off = (my * SZ) % max(1, sz - SZ - 1)
            with open(f, 'rb', buffering=SZ) as fh:
                fh.seek(off); fh.readinto(mv)
    ths = [threading.Thread(target=work) for _ in range(nthreads)]
    t0 = time.perf_counter(); [t.start() for t in ths]; [t.join() for t in ths]
    return nreads*SZ/(time.perf_counter()-t0)/1e9
# 先清 OS 缓存不现实，改用大量不同 offset 覆盖更多数据
for nt in (6, 8, 10):
    gb = run(nt, 300, True)
    print(f'RESULT threads={nt:2d} 300 reads spread: {gb:.2f} GB/s')