
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.config import TinyConfig
from tinyinfer.weights import ShardStore
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
cfg = TinyConfig.from_hf(d)
st = ShardStore(d)
names = [n for n in st.weight_map.keys() if n.endswith(".weight") and "scale" not in n]
print("总 weight 数:", len(names))
# 取第 5 层（有压缩 attention）
import torch
for n in names:
    if n.startswith("layers.5.") and any(k in n for k in ("wq_a","wq_b","wkv","wo_a","wo_b","shared","w1","w2","w3")):
        try:
            w = st.get(n)
            if w.dtype in (torch.float8_e4m3fn,):
                ok = (w.shape[1] % 512 == 0) and (w.shape[0] % 32 == 0)
                print(f"{n.split('layers.5.')[-1]:45s} {str(tuple(w.shape)):18s} {'✓' if ok else '✗ 走旧路径'}")
        except Exception: pass
