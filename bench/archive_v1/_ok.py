
import torch
import tinyinfer._C_stable_libtorch  # noqa
ops = [n for n in torch._C._dispatch_get_all_op_names() if n.startswith("_C::")]
print("ops:", len(ops), "cap:", torch.cuda.get_device_capability())
for n in ("marlin_gemm", "cutlass_mxfp4_group_mm", "mxfp4_experts_quant", "rms_norm"):
    print(" ", n, hasattr(torch.ops._C, n))
