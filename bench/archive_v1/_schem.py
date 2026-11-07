
import torch
import tinyinfer._C_stable_libtorch  # noqa

for n in ("silu_and_mul_with_clamp", "silu_and_mul", "fp32_router_gemm", "dsv3_fused_a_gemm",
          "moe_sum", "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert"):
    ns = torch.ops._C if n != "moe_sum" else torch.ops._moe_C
    try:
        op = getattr(ns, n)
        sch = getattr(op, "_schemas", None)
        print(f"### {n}")
        if sch:
            for s in sch:
                print("   ", str(s)[:260])
        else:
            print("    (no _schemas exposed; opaque registration)")
    except Exception as exc:
        print(f"### {n}: {type(exc).__name__} {str(exc)[:120]}")
