
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())

from tinyinfer.kernels import native
print("native.available('rms_norm'):", native.available("rms_norm", "rms_norm_kernel"))
try:
    native.kernel("rms_norm", "rms_norm_kernel")
    print("kernel resolved (unexpected)")
except Exception as e:
    print("kernel lookup ->", type(e).__name__, str(e)[:130])

from tinyinfer.nn import rms_norm
x = torch.randn(4, 64, dtype=torch.float32)
w = torch.ones(64)
y = rms_norm(x, w, 1e-6)
ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
print("nn.rms_norm ->", tuple(y.shape), "max diff vs torch:", float((y - ref).abs().max()))

# 引擎能否在 CPU 上构造（不加载权重，只看规划器）
try:
    from tinyinfer.resources import probe_hardware, plan
    hw = probe_hardware()
    print("probe ->", hw.name, "vram", round(hw.vram_gb,1), "ram", round(hw.ram_gb,1))
except Exception as e:
    print("probe ->", type(e).__name__, str(e)[:100])
