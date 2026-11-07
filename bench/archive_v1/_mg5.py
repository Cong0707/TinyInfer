import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import moe_gemv
from tinyinfer.kernels.quant import act_quant
n_exp, k, n = 3, 512, 128
lut = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6., -0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], device='cuda')
def fp4(v):
    lo = v[:, 0::2].to(torch.uint8); hi = v[:, 1::2].to(torch.uint8)
    return (lo | (hi << 4)).contiguous()
torch.manual_seed(30)
w1c = torch.randint(0,16,(n_exp,n,k),dtype=torch.uint8,device='cuda')
w3c = torch.randint(0,16,(n_exp,n,k),dtype=torch.uint8,device='cuda')
w1 = [fp4(w1c[e]) for e in range(n_exp)]
w3 = [fp4(w3c[e]) for e in range(n_exp)]
s1 = [torch.full((n,k//32),127,dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
s3 = [torch.full((n,k//32),127,dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
print('RESULT w1[0] shape:', tuple(w1[0].shape), '-> n_rows, khalf')
x = torch.randn(1,k,dtype=torch.bfloat16,device='cuda')
aq, asc = act_quant(x,32,'ue8m0',torch.float8_e8m0fnu)
aq = aq.view(torch.uint8).reshape(-1).contiguous()
asc = asc.view(torch.uint8).reshape(-1).contiguous()
out = moe_gemv.gate_up(aq, asc, w1, s1, w3, s3)
print('RESULT out:', tuple(out.shape), out.dtype)
# 手工参考
af = aq.view(torch.float8_e4m3fn).float().reshape(-1,32)
ascf = torch.pow(2.0,(asc.to(torch.int32)-127).float())
wf = lut[w1c[0].long()].reshape(n,k)
ref = torch.zeros(n, device='cuda')
for kb in range(k//32):
    ref += (wf[:,kb*32:(kb+1)*32] @ af[kb]) * ascf[kb]
print('RESULT ref[:4]:', ref[:4].tolist())
print('RESULT got[:4]:', out[0,:4].float().tolist())