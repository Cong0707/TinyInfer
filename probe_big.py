import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
GB = 2**30
def a():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / GB

from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
print("total %.3fGB" % a(), flush=True)
for k, v in vars(e).items():
    if torch.is_tensor(v) and v.is_cuda:
        print("  %-22s %-14s %.3fGB  %s" % (
              k, str(v.dtype).replace("torch.",""),
              v.numel() * v.element_size() / GB, tuple(v.shape)[:3]), flush=True)
    elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]) and v[0].is_cuda:
        tot = sum(t.numel() * t.element_size() for t in v if torch.is_tensor(t))
        print("  %-22s list[%d] %.3fGB" % (k, len(v), tot / GB), flush=True)
    elif isinstance(v, dict) and v:
        try:
            tot = sum(t.numel() * t.element_size() for t in v.values() if torch.is_tensor(t))
            if tot > 8 * 2**20:
                print("  %-22s dict[%d] %.3fGB" % (k, len(v), tot / GB), flush=True)
        except Exception:
            pass
