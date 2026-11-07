import torch, os
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import gemv_fp4_marlin as GM
dev="cuda"; N,K=2048,7168
w = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device=dev)
s = torch.full((N,K//32),127,dtype=torch.uint8,device=dev)
GM._CACHE.clear(); a = GM._prepared(w,s)
ref = torch.ops._C.gptq_marlin_repack(w.view(torch.uint8).reshape(N,K//2).view(torch.int32).T.contiguous(), K, N, 4, False) if False else None
print('native repack shape:', tuple(a.qweight.shape), a.qweight.dtype)
