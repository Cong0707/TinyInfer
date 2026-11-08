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
base = a()
print("engine (1 layer, no forward) = %.3fGB" % base, flush=True)

# 找最大的几个 tensor
big = []
for obj in list(vars(e).values()):
    if torch.is_tensor(obj) and obj.is_cuda:
        big.append((obj.numel() * obj.element_size() / GB, "attr"))
print("engine-level cuda tensors:", len(big), "sum=%.3fGB" % sum(x for x, _ in big), flush=True)

# 层内 tensor
blk = e._blocks[0]
lsum = 0.0
seen = set()
for owner, name, t in blk._weight_paths():
    if id(t) not in seen:
        seen.add(id(t))
        lsum += t.numel() * t.element_size() / GB
print("layer0 weights = %.3fGB (%d tensors)" % (lsum, len(seen)), flush=True)
print("=> implied fixed overhead = %.3fGB" % (base - lsum), flush=True)
