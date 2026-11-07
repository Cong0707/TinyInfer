
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.BALANCED, device="cuda", layers=list(range(40)))
hw = eng.hw_plan
print("gpu_layers   :", hw.gpu_layers)
print("stream layers:", len(eng._stream_layers()) if hasattr(eng,"_stream_layers") else "?")
print("max_layers   :", eng.max_layers)
print("layer bytes  :", round(hw.layer_bytes/2**20) if hasattr(hw,"layer_bytes") else "?")
p = torch.cuda.get_device_properties(0)
print(f"vram total   : {p.total_memory/2**30:.2f} GiB")
with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=3, temperature=0.0))
    a = torch.cuda.memory_allocated()/2**30
    r_ = torch.cuda.memory_reserved()/2**30
    fr, tot = torch.cuda.mem_get_info()
print(f"allocated    : {a:.2f} GiB   reserved: {r_:.2f} GiB")
print(f"free now     : {fr/2**30:.2f} GiB")
print("host pool GB :", round(eng.host_pool.total/2**30,1) if getattr(eng,"host_pool",None) else 0)
