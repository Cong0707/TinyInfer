
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import ShardIndex
from tinyinfer.resources import model_profile
p = model_profile(ShardIndex(r"E:/models/DeepSeek-V4.1-Flash-1017"))
GB = 1 << 30
for k, v in sorted(p.items()):
    if isinstance(v, (int, float)) and v > 1e6:
        print(f"{k:24s} {v/GB:9.3f} GB")
    else:
        print(f"{k:24s} {v}")
