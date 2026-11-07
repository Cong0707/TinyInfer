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
want = aq.reshape(1, k).view(torch.float8_e4m3fn).float() @ w.float().T
d = (out.float() - want.reshape(-1)).abs()
print('RESULT out[0:4]', out[:4].tolist())
print('RESULT want[0:4]', want[0,:4].tolist())
print('RESULT maxabs', d.max().item(), 'want absmax', want.abs().max().item())