import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import moe_gemv
n_exp, n_rows_out, k_dim = 2, 128, 512
w2 = [torch.randint(0,16,(n_rows_out, k_dim//2),dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
s2 = [torch.full((n_rows_out, k_dim//32),127,dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
aq = torch.randint(0,120,(n_exp, k_dim),dtype=torch.uint8,device='cuda')
asc = torch.full((n_exp, k_dim//32),127,dtype=torch.uint8,device='cuda')
out = moe_gemv.down(aq, asc, w2, s2)
lut = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6., -0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], device='cuda')
# w2[e] 是打包的 [n_rows, k/2]，需按半字节展开成 [n_rows, k]
pk = w2[0]
lo = (pk & 0x0F).long(); hi = ((pk >> 4) & 0x0F).long()
codes = torch.stack((lo, hi), dim=-1).reshape(n_rows_out, k_dim)
wf0 = lut[codes]
af = aq.view(torch.float8_e4m3fn).float()
want0 = af[0] @ wf0.T
print(f'RESULT out[0,:4]: {out[0,:4].float().tolist()}')
print(f'RESULT want[0,:4]: {want0[:4].tolist()}')
d = (out[0,:n_rows_out].float() - want0).abs().max().item()
print(f'RESULT maxdiff: {d:.4e}  rel: {d/want0.abs().max().item():.2e}')