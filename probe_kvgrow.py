"""1M 上下文：压缩 KV 的实际增长与显存占用。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
b = e.block(2)   # 第一个压缩层
a = b.attn
print("layer2: ratio=%s kv_source=%s" % (getattr(a, "ratio", "?"),
      getattr(a, "is_kv_source", "?")), flush=True)
try:
    from tinyinfer.deepseek.compressor import shared_attn as SA
    print("has compress_kv:", getattr(SA, "compress_kv", None) is not None, flush=True)
except Exception as ex:
    print("EXC", str(ex)[:60])

# 跑一个较长的 prefill，看 compress_kv 增长
base = tok.encode("你好，今天天气不错。", add_special_tokens=False)
ids = torch.tensor([(base * 40)[:512]], device="cuda")
print("tokens:", ids.shape[1], flush=True)
e.reset()
torch.cuda.reset_peak_memory_stats()
lg = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
from tinyinfer.deepseek.compressor import shared_attn as SA
ck = getattr(SA, "compress_kv", None)
if ck is not None:
    print("compress_kv shape=%s  %.1f MB" % (tuple(ck.shape),
          ck.numel() * ck.element_size() / 2**20), flush=True)
print("finite=%s  peak=%.0fMB" % (bool(torch.isfinite(lg).all()),
      torch.cuda.max_memory_allocated() / 2**20), flush=True)
