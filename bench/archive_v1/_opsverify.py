
import torch, time
import tinyinfer._C_stable_libtorch, tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer import vllm_ops as V
dev="cuda"
# 确认走 native 而不是 pyd
print("native available rms:", __import__("tinyinfer.native", fromlist=["x"]).available("vllm_rms_norm","vllm_rms_norm_bf16"))
print("native available silu:", __import__("tinyinfer.native", fromlist=["x"]).available("vllm_silu_and_mul","vllm_silu_and_mul_clamp_bf16"))
print("native available topk:", __import__("tinyinfer.native", fromlist=["x"]).available("vllm_topk_softplus_sqrt","vllm_topk_softplus_sqrt_e384_k6"))

x = torch.randn(1, 4096, dtype=torch.bfloat16, device=dev)
w = torch.randn(4096, dtype=torch.bfloat16, device=dev)
ref = torch.empty_like(x)
torch.ops._C.rms_norm(ref, x, w, 1e-6)
got = V.rms_norm(x, w, 1e-6)
print("rms_norm diff:", float((ref-got).abs().max()))

s = torch.randn(1, 4096, dtype=torch.bfloat16, device=dev)
r2 = torch.empty(1, 2048, dtype=torch.bfloat16, device=dev)
torch.ops._C.silu_and_mul_with_clamp(r2, s, 10.0)
g2 = V.clamp_swiglu(s, 10.0)
print("clamp_swiglu diff:", float((r2-g2).abs().max()))

E=384
lg = torch.randn(1, E, dtype=torch.float32, device=dev)
bias = torch.randn(E, dtype=torch.float32, device=dev)
wr = torch.empty(1,6,dtype=torch.float32,device=dev); ir=torch.empty(1,6,dtype=torch.int32,device=dev)
tei=torch.empty(1,6,dtype=torch.int32,device=dev)
torch.ops._moe_C.topk_softplus_sqrt(wr,ir,tei,lg,True,1.5,bias,None,None,None,None,0)
gw, gi = V.gate_topk(lg, 6, True, 1.5, bias)
print("gate_topk idx match:", bool((ir==gi).all()), " wdiff:", float((wr-gw).abs().max()))
