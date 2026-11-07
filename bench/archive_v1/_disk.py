"""What can this disk actually deliver? That number bounds 27B decode."""
import time, os
import ctypes
from pathlib import Path

p = Path(r"E:\Models\Qwen3.8-27B\model-00001-of-00018.safetensors")
size = p.stat().st_size
print(f"test file: {size/2**30:.2f} GB", flush=True)

# plain buffered read
for chunk_mb in (64,):
    chunk = chunk_mb * 2**20
    total = 0
    t0 = time.perf_counter()
    with p.open("rb", buffering=0) as f:
        while total < 2 * 2**30:
            b = f.read(chunk)
            if not b:
                break
            total += len(b)
    dt = time.perf_counter() - t0
    print(f"read {total/2**30:.2f} GB in {dt:.2f}s -> {total/2**30/dt:.2f} GB/s", flush=True)

# random-ish pread of layer-sized blocks (what streaming does)
blocks = [p.open("rb", buffering=0) for _ in range(1)]
fd = os.open(p, os.O_RDONLY | os.O_BINARY)
try:
    off = 0
    total = 0
    t0 = time.perf_counter()
    while total < 2 * 2**30:
        n = 731 * 2**20
        buf = os.pread(fd, min(n, size - off), off)
        if not buf:
            off = 0
            continue
        total += len(buf)
        off += len(buf)
    dt = time.perf_counter() - t0
    print(f"pread 731MB blocks: {total/2**30:.2f} GB in {dt:.2f}s -> "
          f"{total/2**30/dt:.2f} GB/s", flush=True)
finally:
    os.close(fd)
    for f in blocks:
        f.close()
