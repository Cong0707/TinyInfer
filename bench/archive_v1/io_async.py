import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import ShardIndex
from tinyinfer.win_async_io import AsyncFileReader

idx = ShardIndex(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
blobs = [idx.expert_blob(0, e) for e in range(24)]
segs = [(s.file_offset, s.nbytes) for b in blobs for s in b.segments]
gb = sum(n for _, n in segs) / 1e9
print(f"segments={len(segs)} total={gb:.2f}GB", flush=True)
r = AsyncFileReader(idx.model_dir / blobs[0].segments[0].shard, direct=True)
best = 1e9
for _ in range(3):
    t0 = time.time(); outs = r.read_many(segs); dt = time.time() - t0
    best = min(best, dt)
print(f"async OVERLAPPED: {best*1000:.0f}ms {gb/best:.2f}GB/s", flush=True)
r.close()
