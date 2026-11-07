import sys, torch, time, os, glob, threading
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
files = sorted(glob.glob(os.path.join(d,'*.safetensors')))[:4]
SZ = 18_800_000
buf = bytearray(SZ); mv = memoryview(buf)
def run(nthreads, nreads, mode):
    lock = threading.Lock(); st = {'i':0}
    def work():
        while True:
            with lock:
                my = st['i']
                if my >= nreads: return
                st['i'] += 1
            f = files[(my*7) % len(files)]
            sz = os.path.getsize(f)
            off = ((my*37) % 40) * SZ if sz > 40*SZ else (my*SZ) % max(1, sz-SZ-1)
            if mode == 'seek_readinto':
                with open(f,'rb',buffering=0) as fh:
                    fh.seek(off); fh.readinto(mv)
            elif mode == 'buffered':
                with open(f,'rb',buffering=SZ) as fh:
                    fh.seek(off); fh.readinto(mv)
            else:
                fd = os.open(f, os.O_RDONLY | getattr(os,'O_BINARY',0))
                try: os.preadv(fd, [mv], off)
                finally: os.close(fd)
    ths=[threading.Thread(target=work) for _ in range(nthreads)]
    t0=time.perf_counter(); [t.start() for t in ths]; [t.join() for t in ths]
    return nreads*SZ/(time.perf_counter()-t0)/1e9
for mode in ('seek_readinto','buffered','preadv'):
    print(f'RESULT {mode:15s} 16thr: {run(16, 240, mode):.2f} GB/s')