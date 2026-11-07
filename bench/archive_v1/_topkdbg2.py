
import torch
import tinyinfer._C_stable_libtorch, tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer import native
dev="cuda"; E,K = 384,6; T=1
logits = torch.randn(T, E, dtype=torch.float32, device=dev)
bias = torch.zeros(E, dtype=torch.float32, device=dev)
w_m = torch.empty(T,K,dtype=torch.float32,device=dev)
i_m = torch.empty(T,K,dtype=torch.int32,device=dev)
s_m = torch.empty(T,K,dtype=torch.int32,device=dev)

# kernel 签名: (in, out, idx, src_rows, bias, num_rows:int, renormalize:bool, scaling:float)
# 试 1: bool 位置传 int（4字节）
L1 = native.bind("vllm_topk_softplus_sqrt","vllm_topk_softplus_sqrt_e384_k6",(1,),(32,),"TTTTTiif")
L1(w_m, i_m, s_m, logits, bias, T, 1, 1.5); torch.cuda.synchronize()
print("as int  idx:", i_m[0].tolist(), " w:", [round(x,4) for x in w_m[0].tolist()])

# 试 2: bool 位置用 c_bool
w2 = torch.empty_like(w_m); i2 = torch.empty_like(i_m); s2 = torch.empty_like(s_m)
L2 = native.bind("vllm_topk_softplus_sqrt","vllm_topk_softplus_sqrt_e384_k6",(1,),(32,),"TTTTTbif")
L2(w2, i2, s2, logits, bias, T, True, 1.5); torch.cuda.synchronize()
print("as bool idx:", i2[0].tolist(), " w:", [round(x,4) for x in w2[0].tolist()])
