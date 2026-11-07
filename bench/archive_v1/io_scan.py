import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import ShardReader, ShardIndex

idx = ShardIndex(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
blobs = [idx.expert_blob(0, e) for e in range(24)]
gb = sum(sum(s.nbytes for s in b.segments) for b in blobs) / 1e9
for threads in (2, 4, 6, 8, 12, 16):
    try:
        r = ShardReader(idx, threads=threads)
        r.read_blob(blobs[0])
        best = 1e9
        for _ in range(3):
            t0 = time.time(); r.read_many(blobs); dt = time.time() - t0
            best = min(best, dt)
        print(f"threads={threads}: {best*1000:.0f}ms {gb/best:.2f}GB/s", flush=True)
    except Exception as e:
        print(f"threads={threads} ERR {e}", flush=True)
