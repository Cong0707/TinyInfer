import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
st0 = es.stats()
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    s1 = es.stats()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
    s2 = es.stats()
print('RESULT reader:', s2.get('reader'))
print('RESULT stream:', s2.get('stream_reads'), s2.get('stream_gb'))
print('RESULT ram entries:', s2.get('ram',{}).get('entries'), 'hit_rate:', round(s2.get('ram',{}).get('hit_rate',0),3))
delta_bytes = s2['reader']['bytes_read'] - s1['reader']['bytes_read']
print(f'RESULT decode step read {delta_bytes/1e9:.3f} GB from disk')
print(f'RESULT decode step streamed {s2.get("stream_gb",0)-s1.get("stream_gb",0):.3f} GB over PCIe')