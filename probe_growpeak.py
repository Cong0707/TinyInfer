"""触发多次扩容：验证扩容峰值不再翻倍。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[2], tokenizer=tok)
base = tok.encode("你好今天天气不错", add_special_tokens=False)

# 逐步喂，每次都让 compress_kv 有机会增长
ids_all = (base * 400)[:3000]
print("total tokens:", len(ids_all), flush=True)

from tinyinfer.deepseek.compressor import shared_attn as SA
peaks = []
pos = 0
for chunk in (256, 256, 512, 512, 1024, 1024):
    if pos + chunk > len(ids_all):
        break
    seg = torch.tensor([ids_all[pos:pos+chunk]], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    e._forward_chunk(seg, pos)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    ck = getattr(SA, "compress_kv", None)
    sz = 0 if ck is None else ck.numel() * ck.element_size()
    print("pos=%5d chunk=%4d kv=%.1fMB alloc_delta=%+.0fMB peak_delta=%+.0fMB" % (
          pos, chunk, sz/2**20, (torch.cuda.memory_allocated()-before)/2**20,
          (peak-torch.cuda.memory_allocated())/2**20), flush=True)
    pos += chunk
