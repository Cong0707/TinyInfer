
import torch
import tinyinfer._C_stable_libtorch  # noqa
ops = [n for n in torch._C._dispatch_get_all_op_names()
       if n.startswith("_C::") or n.startswith("_moe_C::")]
print("total ops in pyd:", len(ops))
print("cap:", torch.cuda.get_device_capability())
# MLA / grouped 是否在
for name in ("_C::sm100_cutlass_mla_decode", "_C::cutlass_mxfp4_group_mm",
             "_C::mxfp4_experts_quant", "_C::cutlass_scaled_mm", "_C::marlin_gemm"):
    print(f"  {name}: {name in ops}")
