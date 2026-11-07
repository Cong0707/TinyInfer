import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
_dh = []
eng._dump_h = _dh
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    _dh.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
print(f'RESULT dumped {len(_dh)} layers')
if _dh:
    i, hin, hout = _dh[0]
    print(f'RESULT layer {i}: h_in {tuple(hin.shape)} h_out {tuple(hout.shape)} dtype {hout.dtype}')
    print(f'RESULT h_out[-1] shape: {tuple(hout[-1:].shape)}')
# 看 FFN 实际收到什么
b = eng.block(4)
orig = b.ffn.forward
got = []
def w(x, *a, **kw):
    got.append(tuple(x.shape)); return orig(x, *a, **kw)
b.ffn.forward = w
_dh.clear()
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    got.clear()
    eng.forward(torch.tensor([[501]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
print(f'RESULT ffn input shapes: {got[:3]}')