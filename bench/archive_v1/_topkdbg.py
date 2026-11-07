
import torch
import tinyinfer._C_stable_libtorch, tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer import native
dev="cuda"; E,K = 384,6; T=1
logits = torch.randn(T, E, dtype=torch.float32, device=dev)
bias = torch.zeros(E, dtype=torch.float32, device=dev)
w_ref = torch.empty(T,K,dtype=torch.float32,device=dev)
i_ref = torch.empty(T,K,dtype=torch.int32,device=dev)
tei = torch.empty(T,K,dtype=torch.int32,device=dev)
torch.ops._moe_C.topk_softplus_sqrt(w_ref,i_ref,tei,logits,True,1.5,bias,None,None,None,None,0)
print("ref idx:", i_ref[0].tolist())
print("ref w  :", [round(x,4) for x in w_ref[0].tolist()])

w_m = torch.empty_like(w_ref); i_m = torch.empty_like(i_ref); s_m = torch.empty_like(tei)
L = native.bind("vllm_topk_softplus_sqrt","vllm_topk_softplus_sqrt_e384_k6",(1,),(32,),"TTTTTiff")
L(w_m, i_m, s_m, logits, bias, T, True, 1.5)
torch.cuda.synchronize()
print("our idx:", i_m[0].tolist())
print("our w  :", [round(x,4) for x in w_m[0].tolist()])
# 手工算一下正确答案
import torch.nn.functional as F
s = F.softplus(logits.float()).sqrt()
print("manual top6:", torch.topk(s[0], 6).indices.tolist())
