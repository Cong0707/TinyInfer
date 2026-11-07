import sys, builtins, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
real = builtins.__import__
def guard(name, *a, **kw):
    if '_stable_libtorch' in name: raise ImportError('blocked')
    return real(name, *a, **kw)
builtins.__import__ = guard
from tinyinfer.deepseek.fp4_native import gemv_fp4_pq, quantize_act
N, K = 2304, 5120
torch.manual_seed(0)
w = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device='cuda')
s = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
x = torch.randn(1,K,dtype=torch.bfloat16,device='cuda')
aq, asc = quantize_act(x, K)
out = gemv_fp4_pq(aq[0], asc[0], w, s)
print(f'RESULT fp4_native out {tuple(out.shape)} first3={out[0,:3].tolist()}')
# torch 参考
lut = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6., -0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], device='cuda')
lo = (w & 0x0F).long(); hi = ((w>>4)&0x0F).long()
codes = torch.stack((lo,hi),dim=-1).reshape(N,K)
wf = lut[codes]
sc = torch.pow(2.0,(s.to(torch.int32)-127).float()).repeat_interleave(32,0).repeat_interleave(32,1)
deq = (wf*sc)
xf = aq.view(torch.float8_e4m3fn).float() * torch.pow(2.0,(asc.view(torch.uint8).to(torch.int32)-127).float()).repeat_interleave(32)
ref = (deq.float() @ xf.float())
print(f'RESULT torch ref first3={ref[:3].tolist()}')
d=(out[0].float()-ref).abs().max().item()
print(f'RESULT maxdiff={d:.4e}')