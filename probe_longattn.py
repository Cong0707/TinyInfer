"""压缩 attention 的 top-k 索引在长序列上的行为。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[2], tokenizer=tok)
b = e.block(2)
a = b.attn
print("ratio=%s kv_source=%s indexer=%s" % (getattr(a,"ratio","?"),
      getattr(a,"is_kv_source","?"), getattr(a,"indexer",None) is not None), flush=True)
print("index_topk=%s" % (getattr(a, "index_topk", "?"),), flush=True)
base = tok.encode("你好今天天气不错", add_special_tokens=False)
for N in (256, 1024, 2048):
    ids_all = (base * 300)[:N]
    if len(ids_all) < N:
        break
    seg = torch.tensor([ids_all], device="cuda")
    e.reset()
    try:
        lg = e._forward_chunk(seg, 0)
        torch.cuda.synchronize()
        f = lg.float()
        print("N=%4d finite=%-5s absmax=%.2f" % (N, bool(torch.isfinite(f).all()),
              float(f.abs().max()) if torch.isfinite(f).all() else -1), flush=True)
    except Exception as ex:
        print("N=%4d FAIL %s" % (N, str(ex)[:70]), flush=True)
        break
