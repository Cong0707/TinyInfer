import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import ShardReader, ShardIndex

idx = ShardIndex(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
blobs = [idx.expert_blob(0, e) for e in range(6)]
gb = sum(sum(s.nbytes for s in b.segments) for b in blobs) / 1e9
for direct in (True, False):
    try:
        r = ShardReader(idx, threads=8, direct=direct)
        r.read_blob(blobs[0])
        t0 = time.time(); [r.read_blob(b) for b in blobs]; dt = time.time() - t0
        print(f"direct={direct} serial: {dt*1000:.0f}ms {gb/dt:.2f}GB/s", flush=True)
        t0 = time.time(); r.read_many(blobs); dt = time.time() - t0
        print(f"direct={direct} read_many: {dt*1000:.0f}ms {gb/dt:.2f}GB/s", flush=True)
    except Exception as e:
        print(f"direct={direct} ERR {e}", flush=True)
