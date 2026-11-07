
import torch
import tinyinfer._C_stable_libtorch  # noqa
import inspect
f = torch.ops._C.marlin_gemm
print("marlin_gemm op:", f)
# 看 underlying kernel 名
try:
    print("schema:", f._schemas if hasattr(f, "_schemas") else f.default._schema)
except Exception as e:
    print("schema err:", e)
# 对比 gemv_fp4_marlin 里到底调了什么
from tinyinfer import gemv_fp4_marlin as G
print("FP4_TYPE_ID:", G.FP4_TYPE_ID)
print("ops used:", G._ops.__name__ if hasattr(G._ops, "__name__") else G._ops)
