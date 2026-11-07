
import torch
import tinyinfer._C_stable_libtorch, tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer import native
dev="cuda"; E,K = 384,6; T=1
logits = torch.randn(T, E, dtype=torch.float32, device=dev)
bias = torch.zeros(E, dtype=torch.float32, device=dev)
w_m = torch.empty(T,K,dtype=torch.float32,device=dev)
i_m = torch.empty(T,K,dtype=torch.int32,device=dev)
s_m = torch.empty(T,K,dtype=torch.int32,device=dev)
# 正确 kinds: out(T) idx(T) src(T) in(T) bias(T) num_rows(i) renorm(i) scaling(f)
L = native.bind("vllm_topk_softplus_sqrt","vllm_topk_softplus_sqrt_e384_k6",(1,),(32,),"TTTTTiif")
L(w_m, i_m, s_m, logits, bias, T, 1, 1.5); torch.cuda.synchronize()
print("idx:", i_m[0].tolist())
print("w  :", [round(x,4) for x in w_m[0].tolist()])
print("src:", s_m[0].tolist())
