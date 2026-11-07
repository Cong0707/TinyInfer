import vllm._C_stable_libtorch  # noqa
import vllm._moe_C_stable_libtorch  # noqa
import torch
ns = [n for n in dir(torch.ops._C) if not n.startswith("_")]
print("torch.ops._C:", len(ns))
print(ns[:40])
moe = [n for n in dir(torch.ops._moe_C) if not n.startswith("_")] if hasattr(torch.ops, "_moe_C") else []
print("torch.ops._moe_C:", len(moe))
print(moe[:20])