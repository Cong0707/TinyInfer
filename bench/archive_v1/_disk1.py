import sys, os, time, glob
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
files = sorted(glob.glob(os.path.join(d, '*.safetensors')))
print(f'RESULT files: {len(files)}')
# 顺序读大块测上限
tot = 0; t0 = time.perf_counter()
for f in files[:6]:
    with open(f, 'rb', buffering=0) as fh:
        while tot < 3*2**30:
            b = fh.read(16*2**20)
            if not b: break
            tot += len(b)
    if tot >= 3*2**30: break
dt = time.perf_counter()-t0
print(f'RESULT sequential (16MB chunks, direct): {tot/dt/1e9:.2f} GB/s over {tot/2**30:.1f} GB')
# 小块随机读（模拟专家 blob）
tot2 = 0; t0 = time.perf_counter(); n = 0
for f in files[:3]:
    sz = os.path.getsize(f)
    with open(f, 'rb', buffering=0) as fh:
        while n < 150:
            off = (n * 18_800_000) % max(1, sz - 18_800_001)
            fh.seek(off); b = fh.read(18_800_000)
            tot2 += len(b); n += 1
            if b == b'': break
    if n >= 150: break
dt2 = time.perf_counter()-t0
print(f'RESULT 18.8MB random x{n}: {tot2/dt2/1e9:.2f} GB/s')