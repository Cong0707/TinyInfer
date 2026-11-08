import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import ShardIndex
from tinyinfer.resources import model_profile
GB = 2**30
idx = ShardIndex(r"E:\models\DeepSeek-V4.1-Flash-1017")
p = model_profile(idx)
for k, v in sorted(p.items()):
    if isinstance(v, (int, float)) and v > 1e5:
        print("%-18s %10.1f MB   %.3f GB" % (k, v / 2**20, v / GB), flush=True)
    else:
        print("%-18s %s" % (k, v), flush=True)
