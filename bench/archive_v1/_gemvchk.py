
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer
from tinyinfer import moe_gemv
print("available:", moe_gemv.available())
for shape in ((2048, 7168), (7168, 2048)):
    print("  supported", shape, ":", moe_gemv.supported_shape(*shape))
import inspect
src = inspect.getsource(moe_gemv)
print("--- head ---")
print("\n".join(src.splitlines()[:40]))
