
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa

for name in ("cutlass_scaled_mm", "cutlass_scaled_mm_azp", "scaled_fp4_quant", "silu_and_mul", "rms_norm"):
    try:
        print(name, "=>", torch.ops._C.__getattr__(name).default._schema)
    except Exception as ex:
        print(name, "ERR", str(ex)[:80])
print("--- moe ---")
for name in ("moe_sum", "moe_permute", "moe_unpermute"):
    try:
        print(name, "=>", torch.ops._moe_C.__getattr__(name).default._schema)
    except Exception as ex:
        print(name, "ERR", str(ex)[:80])
