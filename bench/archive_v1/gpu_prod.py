import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.loading import ExpertStore
from tinyinfer.weights import ShardStore

torch.manual_seed(0)
model = r"E:\Develop\Deepseek\tinyinfer\cache\weights"
es = ExpertStore.from_store(ShardStore(model), cache_bytes=6 << 30, policy="arc", vram_slots=24, device="cuda", direct=False)
eng = Engine(model, max_layers=1, allow_missing_layers=True, device="cuda", expert_store=es)
print("ready; layers:", eng.available_layers())
ids = [19923, 14, 342, 1030, 22651, 4374, 1465, 16]
with torch.inference_mode():
    eng.forward(torch.tensor([ids], device="cuda"))  # warmup (JIT)
    t0 = time.time()
    for step in range(4):
        eng.forward(torch.tensor([[ids[step]]], device="cuda"), start_pos=step + 1)
    dt = (time.time() - t0) / 4
print(f"decode per step: {dt*1000:.0f} ms (1 layer, GPU)")
st = eng.expert_store.stats()
print("ram:", {k: st["ram"][k] for k in ("hits", "misses", "hit_rate")}, "vram slots:", st.get("vram", {}).get("slots"))
print("vram peak MB:", round(torch.cuda.max_memory_allocated()/2**20))
