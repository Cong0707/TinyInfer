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
            OBS[lid] = torch.full((64,), -999, dtype=torch.int32, device="cuda")
        n = int(indices.numel())
        OBS[lid][:n].copy_(sid[:n])
    return r
MO.MoE._forward_gpu_only = spy
with torch.inference_mode():
    e.forward(ids, 0); e.forward(ids, 0); torch.cuda.synchronize()
    e.pin_hot_experts(); torch.cuda.synchronize()
    e.reset(); e.forward(ids, 1); torch.cuda.synchronize()
    print("GRAPH %.4g" % float(e._graph["out"].abs().max()), flush=True)
    for lid, t in sorted(OBS.items()):
        v = t[:6].tolist()
        print("GSIDS L%d %s neg=%d" % (lid, v, sum(1 for x in v if x < 0)), flush=True)
