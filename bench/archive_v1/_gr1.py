import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    graphs = getattr(eng, '_graphs', None)
    print(f'RESULT _graphs: {len(graphs) if graphs else 0}')
    g_run = eng._graph_run(torch.tensor([[500]], device='cuda'), 8)
    print(f'RESULT _graph_run hit: {g_run is not None}')