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
N, K2 = wg.shape; K = K2*2
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = fp4_native.quantize_act(x, K)
wsg = fp4_native._scale(sg).contiguous(); wsu = fp4_native._scale(su).contiguous()
G = torch.zeros(N, device='cuda', dtype=torch.bfloat16)
U = torch.zeros(N, device='cuda', dtype=torch.bfloat16)
ok = wg.view(torch.uint8).is_contiguous() and wu.view(torch.uint8).is_contiguous()
print('contig', ok, wg.view(torch.uint8).stride(), wu.view(torch.uint8).stride())
native.launch(native.kernel('gemv_fp4_gate_up','gemv_fp4_w4a8_gate_up'), ((N+7)//8,), (256,),
              [aq.reshape(-1), asc.reshape(-1), wg.view(torch.uint8).reshape(-1), wsg.reshape(-1),
               wu.view(torch.uint8).reshape(-1), wsu.reshape(-1), G, U, 1, N, K])
torch.cuda.synchronize()
ref_g = fp4_native.gemv_fp4_pq(aq[0], asc[0], wg, sg)
print('RESULT G', G[:3].tolist(), 'ref', ref_g.flatten()[:3].tolist())
print('RESULT inf count', torch.isinf(G.float()).sum().item(), '/', N)
print('RESULT max finite', G.float()[torch.isfinite(G.float())].abs().max().item() if torch.isfinite(G.float()).any() else 'none')