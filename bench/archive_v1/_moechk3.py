
import torch
from tinyinfer import vllm_ops as vo

vo._ops()
moe = vo._moe_ops()
print("moe ns:", moe)
names = sorted(x for x in dir(moe) if not x.startswith("_"))
print("count:", len(names))
print(names)
