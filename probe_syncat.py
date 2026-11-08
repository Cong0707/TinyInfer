"""逐层同步，看哪一层的 sync 能消除 NaN。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
ids = torch.tensor([tok.encode("1+1 等于几？只回答数字。", add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)

import tinyinfer.deepseek.model as MD
orig = MD.Block.forward
sync_after = int(sys.argv[1]) if len(sys.argv) > 1 else -1
def hooked(self, x, start_pos, pre_mix, positions=None):
    out = orig(self, x, start_pos, pre_mix, positions=positions)
    if sync_after == self.layer_id:
        torch.cuda.synchronize()
    return out
MD.Block.forward = hooked

e.reset()
lg = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
f = lg.float()
print("RESULT sync_after=%d finite=%s absmax=%s nan=%d" % (
      sync_after, bool(torch.isfinite(f).all()),
      float(f.abs().max()) if torch.isfinite(f).all() else "nan",
      int(torch.isnan(f).sum())), flush=True)
