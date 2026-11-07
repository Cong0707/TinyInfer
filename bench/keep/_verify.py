
import torch
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa
ops = [n for n in torch._C._dispatch_get_all_op_names() if n.startswith("_C::") or n.startswith("_moe_C::")]
print("total ops:", len(ops))
print("_C:", len([o for o in ops if o.startswith("_C::")]), "_moe_C:", len([o for o in ops if o.startswith("_moe_C::")]))
print("cap:", torch.cuda.get_device_capability())
for n in ("marlin_gemm", "marlin_gptq_marlin_repack", "gptq_marlin_repack",
          "cutlass_mxfp4_group_mm", "mxfp4_experts_quant", "rms_norm"):
    print(f"  _C::{n}: {('_C::'+n) in ops}")
for n in ("topk_softplus_sqrt", "moe_align_block_size"):
    print(f"  _moe_C::{n}: {('_moe_C::'+n) in ops}")
