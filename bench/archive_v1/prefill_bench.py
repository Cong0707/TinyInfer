import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(24)), allow_missing_layers=True)
print("layers:", eng.available_layers(), "chunk:", eng.prefill_chunk)
for T in (256, 1024, 2048):
    ids = torch.arange(100, 100 + T).unsqueeze(0)
    with torch.inference_mode():
        t0 = time.time()
        logits = eng.forward(ids, 0)
        torch.cuda.synchronize()
        dt = time.time() - t0
    print(f"prefill {T:5d} tok: {dt:.2f}s ({T/dt:.1f} tok/s)  vram {torch.cuda.memory_allocated()/2**20:.0f}MB")
