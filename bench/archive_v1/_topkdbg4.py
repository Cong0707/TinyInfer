
import torch, torch.nn.functional as F
import tinyinfer._C_stable_libtorch, tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer import native
dev="cuda"; E,K = 384,6; T=1
torch.manual_seed(0)
logits = torch.randn(T, E, dtype=torch.float32, device=dev)
bias = torch.zeros(E, dtype=torch.float32, device=dev)
w_m = torch.empty(T,K,dtype=torch.float32,device=dev)
i_m = torch.empty(T,K,dtype=torch.int32,device=dev)
s_m = torch.empty(T,K,dtype=torch.int32,device=dev)
L = native.bind("vllm_topk_softplus_sqrt","vllm_topk_softplus_sqrt_e384_k6",(1,),(32,),"TTTTTiif")
# 正确顺序: input, output, indices, source_rows, bias, num_rows, renorm, scaling
L(logits, w_m, i_m, s_m, bias, T, 1, 1.5); torch.cuda.synchronize()
print("our idx:", i_m[0].tolist())
print("our w  :", [round(x,4) for x in w_m[0].tolist()])
s = F.softplus(logits.float()).sqrt()
v, idx = torch.topk(s[0], 6)
ref_w = (v / v.sum() * 1.5)
print("ref idx:", idx.tolist())
print("ref w  :", [round(x,4) for x in ref_w.tolist()])
