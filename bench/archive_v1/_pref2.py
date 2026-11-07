import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, recursive=False) if False else AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
ids = tok('The quick brown fox jumps over the lazy dog. ' * 5, return_tensors='pt').input_ids[0].tolist()
with torch.inference_mode():
    eng.reset()
    # 预热一次
    eng.forward(torch.tensor([ids], device='cuda'), 0)
    torch.cuda.synchronize()
    s0 = es.stats()
    torch.cuda.synchronize(); t0=time.perf_counter()
    eng.forward(torch.tensor([ids], device='cuda'), 0)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t0)*1000
    s1 = es.stats()
db=(s1['reader']['bytes_read']-s0['reader']['bytes_read'])/1e9
ds=(s1['reader']['read_seconds']-s0['reader']['read_seconds'])*1000
print(f'RESULT prefill({len(ids)} tok): {dt:.0f}ms, disk {db:.2f} GB')
print(f'RESULT disk wall share: {ds/dt*100:.0f}% (累计值，仅供参考)')
print(f'RESULT if zero disk: ~{dt-db/8.8*1000:.0f}ms')