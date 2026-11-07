import vllm._C_stable_libtorch  # noqa
import vllm._moe_C_stable_libtorch  # noqa
import torch
print("attrs:", sorted(torch.ops._C.__dict__.keys())[:30])
for name in ("marlin_gemm", "scaled_fp4_quant", "mxfp4_experts_quant", "nvfp4_blockwise_moe", "gptq_gemm", "awq_gemm", "silu_and_mul", "rms_norm"):
    try:
        op = getattr(torch.ops._C, name)
        print("OK", name, type(op).__name__)
    except Exception as e:
        print("--", name, str(e)[:60])