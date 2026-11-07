import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
# bench.py 的方式：一次 prefill，然后连续 forward 单 token
for n in (1, 8, 11):
    ids = list(range(100, 100+n))
    with torch.inference_mode():
        eng.reset()
        logits = eng.forward(torch.tensor([ids], device='cuda'), 0)
        torch.cuda.synchronize()
        nxt = int(logits[0,-1].argmax())
        pos = len(ids)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            o = eng.forward(torch.tensor([[nxt]], device='cuda'), start_pos=pos)
            torch.cuda.synchronize()
            ts.append(time.perf_counter()-t0)
            nxt = int(o[0,-1].argmax()); pos += 1
        avg = sum(ts[1:])/len(ts[1:])*1000
        print(f'RESULT prompt_len={n:3d} steps={[round(t*1000) for t in ts]} steady_avg={avg:.0f}ms')