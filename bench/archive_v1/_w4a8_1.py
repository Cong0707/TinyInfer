import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.weights import ShardStore
from tinyinfer.deepseek.gemv_fp4 import gemv_fp4_pq, quantize_act, _scale
torch.manual_seed(0)
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
def mine(aq, asc, w, ws):
    n, k2 = w.shape; k = k2*2
    out = torch.empty(n, dtype=torch.bfloat16, device=aq.device)
    nb = k//32
    nblocks = (n + 7)//8
    native.launch(native.kernel('gemv_fp4_w4a8','gemv_fp4_w4a8'), (nblocks,), (256,),
                  [aq.reshape(-1), asc.reshape(-1).float(), w.view(torch.uint8).reshape(-1),
                   ws.reshape(-1), out, 1, n, k])
    return out.view(1, n)
for ni in (0, 1, 34):
    nm = f'layers.3.ffn.experts.{ni}.w1'
    w = st.get(nm + '.weight').to('cuda')
    g = st.get(nm + '.scale').to('cuda')
    K = w.shape[1]*2
    x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
    aq, asc = quantize_act(x, K)
    aq = aq.reshape(-1); asc = asc.reshape(-1)
    ws = _scale(g)
    n = w.shape[0]
    if ws.size(0) != n:
        ws = ws.view(-1, ws.size(-1)).repeat_interleave(32, 0)[:n]
    ref = gemv_fp4_pq(aq, asc, w, g, ws)
    got = mine(aq, asc, w, ws)
    a = ref.float().flatten(); b = got.float().flatten()
    print(f'RESULT expert{ni}: maxdiff={(a-b).abs().max().item():.3e} rel={(a-b).abs().max().item()/max(1e-6,a.abs().max().item()):.2e}')