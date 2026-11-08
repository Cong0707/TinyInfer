"""二分 reset()：跳过哪一段能避免 NaN。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

SKIP = sys.argv[1]
import tinyinfer.engine as E
orig_reset = E.Engine.reset

def patched(self):
    if SKIP == "attn":
        for b in self._blocks.values():
            pass
        # 跳过 attention 的 reset
        for _a in ("_flat_ev",):
            if hasattr(self, _a):
                getattr(self, _a).clear()
        for b in self._blocks.values():
            b.ffn._expert_cache.clear()
        return
    if SKIP == "blocks":
        return orig_reset(self)
    return orig_reset(self)
E.Engine.reset = patched

M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
L = torch.tensor([tok.encode("1+1 等于几？只回答数字。", add_special_tokens=False)], device="cuda")
S = torch.tensor([tok.encode("你好", add_special_tokens=False)], device="cuda")

e.reset()
e._forward_chunk(L, 0)
torch.cuda.synchronize()
e.reset()
lg = e._forward_chunk(S, 0)
torch.cuda.synchronize()
print("SKIP=%-8s finite=%s" % (SKIP, bool(torch.isfinite(lg).all())), flush=True)
