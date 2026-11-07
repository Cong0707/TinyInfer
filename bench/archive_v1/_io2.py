import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.loading import ShardIndex, ShardReader
import os, glob
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
idx = ShardIndex(d)
sr = ShardReader(idx, threads=8)
# 读一批从未读过的专家（不同层不同专家），追踪真实耗时
tot_b = 0; tot_t = 0.0
n_ok = 0
for layer in (20, 21, 22, 23):
    for e in (100, 150, 200, 250, 300, 350):
        try:
            blob = idx.expert_blob(layer, e)
            t0 = time.perf_counter()
            data = sr.read_blob(blob)
            dt = time.perf_counter() - t0
            tot_b += len(data); tot_t += dt; n_ok += 1
        except Exception as ex:
            break
print(f'RESULT cold-ish reads: {n_ok} blobs, {tot_b/1e9:.2f} GB, {tot_t*1000:.0f} ms wall')
print(f'RESULT effective (serial): {tot_b/tot_t/1e9:.2f} GB/s')
print(f'RESULT per blob: {tot_t/n_ok*1000:.1f} ms')