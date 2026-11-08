import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.config import TinyConfig
cfg = TinyConfig.from_hf(r"E:\models\DeepSeek-V4.1-Flash-1017")
r = list(cfg.compress_ratios)
print("n=%d" % len(r))
print("ratios:", r)
from collections import Counter
print("count:", dict(Counter(r)))
# 1M 上下文的压缩 KV
head_dim = cfg.head_dim
total = 0
for x in r:
    if x > 0:
        total += (1_000_000 // x) * head_dim // 2
print("1M KV = %.2f GB" % (total / 2**30))
