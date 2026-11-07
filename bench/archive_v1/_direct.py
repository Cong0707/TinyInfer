import sys, time, threading
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import _WindowsDirect, ShardIndex
idx = ShardIndex(r"E:\models\DeepSeek-V4.1-Flash-1017")
N = 18800640
buf = bytearray(N)
def one(d, path, off):
    d.read(path, off, N, buf)
paths = idx.shard_paths if hasattr(idx, "shard_paths") else None
print("shards:", len(idx.shards) if hasattr(idx, "shards") else "?")
import glob
files = sorted(glob.glob(r"E:\models\DeepSeek-V4.1-Flash-1017\*.safetensors"))
# 单线程
d = _WindowsDirect(max_scratch=32<<20)
t0=time.perf_counter()
for i in range(6): one(d, files[i], 10_000_000)
dt=time.perf_counter()-t0
print(f"direct 单线程 x6: {6*N/1e9/dt:.2f} GB/s  ({dt*1000:.0f}ms)")
# 6 并发
ds = [_WindowsDirect(max_scratch=32<<20) for _ in range(6)]
ths=[threading.Thread(target=one,args=(ds[i],files[i],10_000_000)) for i in range(6)]
t0=time.perf_counter()
for th in ths: th.start()
for th in ths: th.join()
dt=time.perf_counter()-t0
print(f"direct 6并发:    {6*N/1e9/dt:.2f} GB/s  ({dt*1000:.0f}ms)")
