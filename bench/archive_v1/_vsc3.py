import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.loading import VramSlotCache

blob = 1 << 20  # 1MB
c = VramSlotCache(4, blob, device='cuda:0')
print(f'RESULT slots={c.slots} blob={c.blob_bytes} device={c.device}')
print(f'RESULT stats: {c.stats()}')

# acquire 一个 slot
key = (0, 42)
try:
    s = c.acquire(key)
    print(f'RESULT acquire({key}) -> {s}')
    print(f'RESULT has({key}) = {c.has(key)}')
    buf = c.buffer(s)
    print(f'RESULT buffer shape {tuple(buf.shape)} dtype {buf.dtype}')
    c.release(s)
    print(f'RESULT after release, staged_count={c.staged_count()}')
except Exception as e:
    import traceback; traceback.print_exc()