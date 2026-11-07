import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.deepseek import fp4_native
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
L = 3
sg = st.get(f'layers.{L}.ffn.experts.0.w1.scale').to('cuda')
wg = st.get(f'layers.{L}.ffn.experts.0.w1.weight').to('cuda')
print('w shape', tuple(wg.shape), 'scale shape', tuple(sg.shape), 'dtype', sg.dtype)
N = wg.shape[0]; K = wg.shape[1]*2
ws = fp4_native._scale(sg)
print('after _scale:', tuple(ws.shape), ws.dtype)
print('N=', N, 'expect [N, K/32] =', (N, K//32))
if ws.size(0) != N:
    ws2 = ws.view(-1, ws.size(-1)).repeat_interleave(32, 0)[:N]
    print('prep ->', tuple(ws2.shape))