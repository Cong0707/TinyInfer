import sys, time, threading
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.loading import ShardIndex, ShardReader
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
idx = ShardIndex(d)
def run(nthreads, blobs):
    sr = ShardReader(idx, threads=nthreads)
    lock = threading.Lock(); st = {'i': 0}; got = [0]
    def work():
        while True:
            with lock:
                my = st['i']
                if my >= len(blobs): return
                st['i'] += 1
            try:
                b = idx.expert_blob(*blobs[my])
                d_ = sr.read_blob(b)
                with lock: got[0] += len(d_)
            except Exception: pass
    ths = [threading.Thread(target=work) for _ in range(nthreads)]
    t0 = time.perf_counter(); [t.start() for t in ths]; [t.join() for t in ths]
    return got[0]/(time.perf_counter()-t0)/1e9, got[0]/1e9
# 用不同层的专家，尽量减少页缓存影响
blobs = [(L, e) for L in (25, 26, 27, 28, 29) for e in (50, 120, 200, 280, 360)]
for nt in (1, 4, 8):
    gb, tot = run(nt, blobs)
    print(f'RESULT {nt} threads: {gb:.2f} GB/s over {tot:.2f} GB')