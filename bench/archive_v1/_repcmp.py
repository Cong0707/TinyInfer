
import torch
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import gemv_fp4_marlin as GM
dev="cuda"; N,K=2048,7168
w = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device=dev)
s = torch.full((N,K//32),127,dtype=torch.uint8,device=dev)
packed = w.reshape(N, K//2).contiguous().view(torch.uint8).view(torch.int32).T.contiguous()
ref = torch.ops._C.gptq_marlin_repack(packed, K, N, 4, False)
got = GM._repack(packed, K, N, torch.ops._C)
torch.cuda.synchronize()
print("shapes:", tuple(ref.shape), tuple(got.shape))
print("bit-identical:", bool((ref == got).all()))
if not bool((ref==got).all()):
    print("diff count:", int((ref != got).sum()))
