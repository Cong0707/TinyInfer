import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import moe_gemv
# down: w2[e] 形状 (n_rows_out, k_dim/2)，kernel 要求 k_dim % 256 == 0
# 真实模型: w2 是 [5120, 1152] (dim=5120, moe_inter/2=1152) -> k_dim=2304
# 这里用小型但满足约束的: n_rows_out=128, k_dim=512 -> w2 shape (128, 256)
n_exp, n_rows_out, k_dim = 2, 128, 512
w2 = [torch.randint(0,16,(n_rows_out, k_dim//2),dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
s2 = [torch.full((n_rows_out, k_dim//32),127,dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
aq = torch.randint(0,120,(n_exp, k_dim),dtype=torch.uint8,device='cuda')
asc = torch.full((n_exp, k_dim//32),127,dtype=torch.uint8,device='cuda')
out = moe_gemv.down(aq, asc, w2, s2)
print(f'RESULT down out: {tuple(out.shape)} {out.dtype}')
print(f'RESULT expect: (n_exp={n_exp}, 2*n_rows={2*n_rows_out})')
lut = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6., -0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], device='cuda')
af = aq.view(torch.float8_e4m3fn).float()
wf0 = lut[w2[0].long()]
want0 = af[0] @ wf0.T
print(f'RESULT out[0,:4]: {out[0,:4].float().tolist()}')
print(f'RESULT want[0,:4]: {want0[:4].tolist()}')