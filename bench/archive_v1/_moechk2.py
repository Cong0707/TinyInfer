
import torch
from tinyinfer import vllm_ops as vo
vo.available()

def probe(ns_name, op_name):
    ns = getattr(torch.ops, ns_name, None)
    if ns is None:
        return f"namespace {ns_name} missing"
    try:
        op = getattr(ns, op_name)
        return f"OK  {op}"
    except Exception as exc:
        return f"MISSING ({type(exc).__name__}: {exc})"

for ns, names in {
    "_moe_C": ["topk_softplus_sqrt", "grouped_topk", "topk_softmax",
               "moe_align_block_size", "moe_wna16_marlin_gemm", "moe_sum",
               "moe_permute", "moe_unpermute", "marlin_gemm"],
    "_C": ["marlin_gemm", "moe_align_block_size", "moe_sum", "moe_permute",
           "moe_unpermute", "silu_and_mul"],
}.items():
    for n in names:
        print(f"{ns}.{n}: {probe(ns, n)}")
print()
print("moe_C dir sample:", [x for x in dir(torch.ops._moe_C) if not x.startswith("_")][:20])
