import vllm._C_stable_libtorch  # noqa
import torch
for name in ("marlin_gemm", "scaled_fp4_quant", "mxfp4_experts_quant", "silu_and_mul", "rms_norm", "gptq_gemm", "awq_gemm"):
    op = getattr(torch.ops._C, name)
    for ov in op.overloads():
        print(name + "." + ov + ": " + str(getattr(op, ov)._schema)[:300])
        break