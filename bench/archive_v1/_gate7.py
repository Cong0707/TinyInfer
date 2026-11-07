import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=[0])
g = eng.block(0).ffn.gate
print('RESULT topk', g.topk, 'cfg.n_activated', eng.cfg.n_activated_experts)
print('RESULT norm', g.norm_topk_prob, 'scale', g.route_scale, 'score', g.score_func)
print('RESULT bias_vl', g.bias_vl is not None)