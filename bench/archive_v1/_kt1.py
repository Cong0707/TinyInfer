import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
torch.manual_seed(11)
n, k = 256, 512
w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device='cuda')
ws = torch.rand(n, k // 32, device='cuda') + 0.5
aq = torch.randint(0, 120, (k,), dtype=torch.uint8, device='cuda')
asc = (torch.rand(k // 32, device='cuda') * 0.2 + 0.01).to(torch.float32)
out = torch.empty(n, dtype=torch.bfloat16, device='cuda')
print('RESULT aq dtype', aq.dtype, 'shape', tuple(aq.shape))
print('RESULT w dtype', w.dtype, 'shape', tuple(w.shape))
print('RESULT ws shape', tuple(ws.shape))
try:
    native.launch(native.kernel('gemv_fp4_w4a8','gemv_fp4_w4a8'), ((n+7)//8,), (256,),
                  [aq, asc, w.reshape(-1), ws.reshape(-1), out, 1, n, k])
    torch.cuda.synchronize()
    print('RESULT out[:3]', out[:3].tolist(), 'absmax', out.float().abs().max().item())
    print('RESULT ws absmax', ws.abs().max().item(), 'asc', asc[:3].tolist())
except Exception as e:
    print('RESULT FAIL', repr(e)[:150])