import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
import torch

eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(40)))
t0 = time.time()
for i in range(40):
    eng.block(i)
print(f"40 block construction: {(time.time()-t0):.1f}s")
with torch.inference_mode():
    t0 = time.time()
    eng.forward(torch.tensor([[1, 2, 3, 4]]), 0)
    torch.cuda.synchronize()
    print(f"first forward (incl JIT): {(time.time()-t0):.1f}s")
    t0 = time.time()
    eng.forward(torch.tensor([[1, 2, 3, 4]]), 4)
    torch.cuda.synchronize()
    print(f"second forward (warm): {(time.time()-t0):.1f}s")
