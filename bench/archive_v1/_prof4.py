
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import ShardIndex
idx = ShardIndex(r"E:/models/DeepSeek-V4.1-Flash-1017")
hits = [n for n in idx.weight_map if n.endswith("embed.weight") or n.endswith("embed.scale")]
for h in sorted(hits)[:6]:
    print(repr(h))
print("total matches:", len(hits))
