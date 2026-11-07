import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import moe_gemv
n_exp, k, n = 3, 512, 256
print('RESULT === acc 测试 ===')
src = torch.randn(n_exp, n, dtype=torch.bfloat16, device='cuda')
y0 = torch.randn(n, dtype=torch.bfloat16, device='cuda')
y = y0.clone()
moe_gemv.acc(src, y, n_exp)
torch.cuda.synchronize()
want = y0.float() + src.float().sum(0)
print(f'RESULT y[:3]={y[:3].float().tolist()}')
print(f'RESULT want[:3]={want[:3].tolist()}')
print(f'RESULT any nan: {torch.isnan(y.float()).any().item()}')
print('RESULT === down 形状 ===')
w2c = torch.randint(0,16,(n_exp,k,n),dtype=torch.uint8,device='cuda')
print(f'RESULT w2c[0]: {tuple(w2c[0].shape)} = (k={k}, n={n})')
print('RESULT 这意味着 down 的 n_rows=%d, khalf=%d, k_dim=%d' % (k, n//2, n))
print(f'RESULT k_dim={n} 需要 %%256==0 -> {n%256==0}')