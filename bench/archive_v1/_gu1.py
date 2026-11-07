import sys, torch, time
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
wsg = fp4_native._scale(sg); wsu = fp4_native._scale(su)
def prep(ws):
    return ws.view(-1, ws.size(-1)).repeat_interleave(32, 0)[:N] if ws.size(0) != N else ws
wsg = prep(wsg); wsu = prep(wsu)
# 分别算
ref_g = fp4_native.gemv_fp4_pq(aq[0], asc[0], wg, sg)
ref_u = fp4_native.gemv_fp4_pq(aq[0], asc[0], wu, su)
# 融合
G = torch.empty(N, device='cuda', dtype=torch.bfloat16)
U = torch.empty(N, device='cuda', dtype=torch.bfloat16)
wb_g = wg.view(torch.uint8).reshape(-1); wb_u = wu.view(torch.uint8).reshape(-1)
a1 = aq.reshape(-1); s1 = asc.reshape(-1)
native.launch(native.kernel('gemv_fp4_gate_up','gemv_fp4_gate_up'), ((N+7)//8,), (256,),
              [a1, s1, wb_g, wsg.reshape(-1), wb_u, wsu.reshape(-1), G, U, 1, N, K])
torch.cuda.synchronize()
print('RESULT gate diff', (ref_g.flatten().float()-G.float()).abs().max().item())
print('RESULT up   diff', (ref_u.flatten().float()-U.float()).abs().max().item())
def bench(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e6
t_two = bench(lambda: (fp4_native.gemv_fp4_pq(aq[0], asc[0], wg, sg), fp4_native.gemv_fp4_pq(aq[0], asc[0], wu, su)))
t_fuse = bench(lambda: native.launch(native.kernel('gemv_fp4_gate_up','gemv_fp4_gate_up'), ((N+7)//8,), (256,), [a1, s1, wb_g, wsg.reshape(-1), wb_u, wsu.reshape(-1), G, U, 1, N, K]))
print(f'RESULT two={t_two:.1f}us fused={t_fuse:.1f}us speedup={t_two/t_fuse:.2f}x')