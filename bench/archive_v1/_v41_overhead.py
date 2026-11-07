import json, struct
from pathlib import Path
from collections import defaultdict
MB = 2**20; GB = 2**30

MD = Path(r"E:\Models\DeepSeek-V4.1-Flash-1017")
idx = json.loads((MD/"model.safetensors.index.json").read_text(encoding="utf-8"))
wm = idx["weight_map"]; hd = {}
def hdr(s):
    if s not in hd:
        with (MD/s).open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hd[s] = json.loads(f.read(n))
    return hd[s]

print("=== V4.1 独有的开销：engram 和超连接 ===")
print()
g = defaultdict(int)
for name, shard in wm.items():
    e = hdr(shard).get(name)
    if e is None: continue
    b = e["data_offsets"][1]-e["data_offsets"][0]
    if "engram" in name: g["engram"] += b
    elif "hc_" in name: g["hyper_connections"] += b
    elif "compressor" in name: g["compressor"] += b
    elif "indexer" in name: g["indexer"] += b
for k, b in sorted(g.items(), key=lambda kv:-kv[1]):
    print(f"  {k:20s} {b/MB:9.1f} MB")
print()
print("engram 只有 2 层（layer 1, 14）有，但每层是巨型查表：")
print("  每 token 要做 n-gram 查找 + 投影，可能引入额外延迟")
print()
print("=== 每 token 的额外操作（R1 没有的）===")
print("  1. engram n-gram 查找（2 层）")
print("  2. 超连接 hc_pre/hc_post（40 层 x 4 路）")
print("  3. 压缩注意力的 indexer 两级索引")
print("  4. compressor 在线压缩")
print()
print("这些都是小算子，但【每层多几次 kernel】= 40 层 x 额外 5-10 个 kernel")
print("= 200-400 次额外 launch，在 decode 时是实实在在的开销。")