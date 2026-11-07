import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.kernels.quant import act_quant
torch.manual_seed(2)
n, k = 512, 512
w = torch.randn(n, k, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
ws = torch.full((n//32, k//32), 127, dtype=torch.uint8, device='cuda')
x = torch.randn(1, k, dtype=torch.bfloat16, device='cuda')
aq, asc = act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
out = torch.empty(n, device='cuda', dtype=torch.bfloat16)
native.launch(native.kernel('fp8_gemv_bs','fp8_gemv_bs'), ((n+15)//16,), (512,),
              [aq.view(torch.uint8).reshape(-1), asc.view(torch.uint8).reshape(-1),
               w.view(torch.uint8).reshape(-1), ws.view(-1), out, n, k])
torch.cuda.synchronize()
# 正确参考：per-32-block scale 内部求和
Af = aq.reshape(1, k).view(torch.float8_e4m3fn).float()
Wf = w.float()
asf = torch.pow(2.0, (asc.view(torch.uint8).to(torch.int32)-127).to(torch.float32)).reshape(-1)
wsf = torch.pow(2.0, (ws.to(torch.int32)-127).to(torch.float32))
ref = torch.zeros(n, device='cuda')
for kb in range(k//32):
    p = (Af[0, kb*32:(kb+1)*32] * Wf[:, kb*32:(kb+1)*32]).sum(dim=1)
    wsc = wsf[(torch.arange(n, device='cuda')//32), kb]
    ref += p * asf[kb] * wsc
d = (out.float() - ref).abs()
print('RESULT maxabs', d.max().item(), 'refabsmax', ref.abs().max().item(), 'rel', (d.max()/ref.abs().max()).item())
print('RESULT out[:3]', out[:3].tolist())
print('RESULT ref[:3]', ref[:3].tolist())