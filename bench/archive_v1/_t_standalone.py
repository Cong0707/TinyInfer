
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch  # must come first: torch's __init__ registers the DLL search dirs
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa
print("standalone load OK")
print("marlin_gemm:", hasattr(torch.ops._C, "marlin_gemm"))
x = torch.randn(1, 5120, dtype=torch.bfloat16, device="cuda")
w = torch.randn(5120, dtype=torch.bfloat16, device="cuda")
out = torch.empty_like(x)
torch.ops._C.rms_norm(out, x, w, 1e-6)
print("rms_norm ran, finite:", bool(torch.isfinite(out).all()))
