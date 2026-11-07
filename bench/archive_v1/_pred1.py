import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
# 检查 gate 是否可直接调用
b = eng.block(3)
print(f'RESULT block3 ffn: {type(b.ffn).__name__}')
print(f'RESULT has gate: {hasattr(b.ffn, "gate")}')
g = b.ffn.gate
print(f'RESULT gate type: {type(g).__name__}')
print(f'RESULT gate weight: {getattr(g, "weight", None) is not None}')
# 试：给定一个 hidden，能否算出路由
x = torch.randn(1, 5120, dtype=torch.bfloat16, device='cuda')
try:
    out = g(x)
    print(f'RESULT gate output: {[tuple(t.shape) for t in out]}')
    print(f'RESULT topk experts: {out[1].tolist()}')
except Exception as e:
    print(f'RESULT gate FAILED: {repr(e)[:120]}')