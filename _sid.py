import sys, torch
sys.path.insert(0, "/models/tinyinfer/src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = "/models/deepseek"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
pad = tok.encode("你好", add_special_tokens=False)[0]
ids = torch.tensor([[pad]], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1], tokenizer=tok)

OBS = {}
def make_obs(name, n):
    t = torch.full((64,), -999, dtype=torch.int32, device="cuda")
    OBS[name] = t
    return t

# 在每个 MoE 的 _forward_gpu_only 入口处，把 sids 拷进 OBS（图内 copy_，合法）
import tinyinfer.deepseek.moe as MO
orig = MO.MoE._forward_gpu_only
def spy(self, x, xf, indices, weights):
    lid = self.layer_id
    r = orig(self, x, xf, indices, weights)
    sid = getattr(self, "_sids_buf", None)
    if sid is not None:
        if lid not in OBS:
            if torch.cuda.is_current_stream_capturing():
                return r
            OBS[lid] = make_obs(lid, 64)
        try:
            n = int(indices.numel())
            OBS[lid][:n].copy_(sid[:n])
        except Exception:
            pass
    return r
MO.MoE._forward_gpu_only = spy

with torch.inference_mode():
    e.forward(ids, 0); e.forward(ids, 0); torch.cuda.synchronize()
    e.pin_hot_experts(); torch.cuda.synchronize()
    e.reset(); e.forward(ids, 1); torch.cuda.synchronize()
    print("GRAPH %.4g" % float(e._graph["out"].abs().max()), flush=True)
    for lid, t in sorted(OBS.items()):
        v = t[:6].tolist()
        print("GRAPH-SIDS L%d %s" % (lid, v), flush=True)
    OBS.clear()
    e.reset(); e._forward_impl(ids, 1); torch.cuda.synchronize()
    for lid, t in sorted(OBS.items()):
        v = t[:6].tolist()
        print("EAGER-SIDS L%d %s" % (lid, v), flush=True)
