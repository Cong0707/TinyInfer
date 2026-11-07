import sys, json, pathlib
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
d = pathlib.Path(r'E:\\models\\DeepSeek-V4.1-Flash-1017')
idx = json.load(open(d/'model.safetensors.index.json'))
wm = idx['weight_map']
# 找 mtp / dspark 相关的 key
mtp_keys = sorted(k for k in wm if 'mtp' in k.lower() or 'dspark' in k.lower() or 'nextn' in k.lower())
print(f'RESULT mtp keys: {len(mtp_keys)}')
for k in mtp_keys[:8]: print('  ', k)
# 哪些 shard 装 mtp
shards = sorted(set(wm[k] for k in mtp_keys))
print(f'RESULT mtp shards: {shards}')
tot = 0
for s in shards:
    sz = (d/s).stat().st_size
    tot += sz
    print(f'  {s} {sz/2**20:.0f} MB')
print(f'RESULT mtp total {tot/2**20:.0f} MB')