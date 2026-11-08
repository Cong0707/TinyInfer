"""40 层时显存到底被谁占了：初始化后、forward 前的快照。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

def rpt(tag):
    f, t = torch.cuda.mem_get_info()
    print("%-22s free=%5.0fMB total=%5.0fMB alloc=%5.0fMB reserved=%5.0fMB" % (
        tag, f / 2**20, t / 2**20,
        torch.cuda.memory_allocated() / 2**20,
        torch.cuda.memory_reserved() / 2**20), flush=True)

rpt("before engine")
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda",
           layers=list(range(40)), tokenizer=tok)
rpt("after engine init")
sl = e.expert_store.slots
print("slots:", None if sl is None else (sl.slots, sl.blob_bytes), flush=True)
try:
    sl._ensure_arena() if hasattr(sl, "_ensure_arena") else None
except Exception as ex:
    print("arena:", str(ex)[:100], flush=True)
rpt("after arena")
