
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import ShardIndex
from tinyinfer.adapters import open_adapter
a = open_adapter(r"E:\models\DeepSeek-V4.1-Flash-1017")
idx = ShardIndex(a.store.model_dir)
wm = idx.weight_map
hdr = idx._header
GB = 2**30

# 按 tensor 名归类一层的构成
import re, collections
cat = collections.Counter()
for name, shard in wm.items():
    m = re.match(r"layers\.(\d+)\.(.+)$", name)
    if not m: continue
    if int(m.group(1)) != 0: continue
    rest = m.group(2)
    try:
        shapes = hdr.get(name)
    except Exception:
        shapes = None
    n = 0
    # 用文件大小无法分 tensor；退而求其次：按数量统计
    cat[rest.split(".")[0]] += 1
print("layer0 tensor 分组(count):")
for k, v in cat.most_common(12):
    print(f"  {k:<18} {v} tensors")
