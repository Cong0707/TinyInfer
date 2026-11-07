
import torch
import tinyinfer._C_stable_libtorch  # noqa
print("torch", torch.__version__, "cap", torch.cuda.get_device_capability())
for fn in ("cutlass_scaled_mm_supports_fp4", "cutlass_group_gemm_supported"):
    op = getattr(torch.ops._C, fn)
    print(f"  {fn}(120) =", op(120))
print("mxfp4 op present:", getattr(torch.ops._C, "cutlass_mxfp4_group_mm", None) is not None)
