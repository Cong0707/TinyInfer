import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.deepseek import fp4_native
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
L = 3
wg = st.get(f'layers.{L}.ffn.experts.0.w1.weight').to('cuda')
sg = st.get(f'layers.{L}.ffn.experts.0.w1.scale').to('cuda')
wu = st.get(f'layers.{L}.ffn.experts.0.w3.weight').to('cuda')
su = st.get(f'layers.{L}.ffn.experts.0.w3.scale').to('cuda')
N, K2 = wg.shape; K = K2*2; nb = K//32
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = fp4_native.quantize_act(x, K)
wsg = fp4_native._scale(sg).contiguous()
wsu = fp4_native._scale(su).contiguous()
print('wsg', tuple(wsg.shape), 'wsu', tuple(wsu.shape))
G = torch.full((N,), -777.0, device='cuda', dtype=torch.bfloat16)
U = torch.full((N,), -888.0, device='cuda', dtype=torch.bfloat16)
wb_g = wg.view(torch.uint8).reshape(-1); wb_u = wu.view(torch.uint8).reshape(-1)
try:
    native.launch(native.kernel('gemv_fp4_gate_up','gemv_fp4_w4a8_gate_up'), ((N+7)//8,), (256,),
                  [aq.reshape(-1), asc.reshape(-1), wb_g, wsg.reshape(-1), wb_u, wsu.reshape(-1), G, U, 1, N, K])
    torch.cuda.synchronize()
    print('G sample', G[:4].tolist(), 'nan:', torch.isnan(G.float()).any().item())
    print('U sample', U[:4].tolist(), 'nan:', torch.isnan(U.float()).any().item())
    print('unwritten G:', (G.float()==-777.0).sum().item(), '/', N)
except Exception as e:
    print('FAIL', repr(e)[:150])