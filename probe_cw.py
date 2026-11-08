"""cold vs warm 在 2048 时的差异：逐层逐组件。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("你好今天天气不错", add_special_tokens=False)

mode = sys.argv[1]
if mode == "warm":
    for w in (64, 256, 512, 1024):
        e.reset()
        e._forward_chunk(torch.tensor([(base*2000)[:w]], device="cuda"), 0)
    torch.cuda.synchronize()

rec = []
import tinyinfer.deepseek.model as MD
orig = MD.Block.forward
def hook(self, x, start_pos, pre_mix, positions=None):
    o = orig(self, x, start_pos, pre_mix, positions=positions)
    h = o[0] if isinstance(o, tuple) else o
    rec.append((self.layer_id, bool(torch.isfinite(h.float()).all()),
                float(h.float().abs().max()) if torch.isfinite(h.float()).all() else -1))
    return o
MD.Block.forward = hook

e.reset()
lg = e._forward_chunk(torch.tensor([(base*2000)[:2048]], device="cuda"), 0)
torch.cuda.synchronize()
print("RESULT mode=%s finite=%s" % (mode, bool(torch.isfinite(lg).all())), flush=True)
for lid, ok, am in rec:
    print("  L%d finite=%-5s absmax=%.3f" % (lid, ok, am), flush=True)
