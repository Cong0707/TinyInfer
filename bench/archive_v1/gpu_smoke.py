import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine

torch.manual_seed(0)
print("cuda:", torch.cuda.is_available())
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", max_layers=1, allow_missing_layers=True, device="cuda")
print("layers:", eng.available_layers())
ids = [19923, 14, 342, 1030, 22651, 4374, 1465, 16]
t0 = time.time()
with torch.inference_mode():
    logits = eng.forward(torch.tensor([ids], device="cuda"))
print("prefill", tuple(logits.shape), "finite", torch.isfinite(logits).all().item(), f"{time.time()-t0:.1f}s")
t0 = time.time()
out = eng.generate(ids, max_new_tokens=3, temperature=0.0)
print("gen", out, f"{time.time()-t0:.2f}s")
print("vram peak MB:", round(torch.cuda.max_memory_allocated()/2**20))
