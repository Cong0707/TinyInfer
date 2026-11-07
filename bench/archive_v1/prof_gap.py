import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(26)))
with torch.inference_mode():
    eng.forward(torch.tensor([[1, 2, 3, 4]]), 0)
    nid = int(eng.forward(torch.tensor([[1]]), 4)[0, -1].argmax())
    torch.cuda.synchronize()
    # 用 forward 的 profile events（每层 4 段）+ 层间间隙
    for i in eng.layer_ids:
        eng.block(i).profile = True
    eng.profile = True
    t0 = time.perf_counter()
    eng.forward(torch.tensor([[nid]]), 5)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000
    parts = {i: dict(eng.block(i)._parts) for i in eng.layer_ids}
gpu = sum(sum(p.values()) for p in parts.values())
print(f"wall {wall:.0f}ms  gpu_kernel_sum {gpu:.0f}ms  unaccounted {wall-gpu:.0f}ms ({(wall-gpu)/wall*100:.0f}%)")
