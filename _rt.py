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

R = {}
import tinyinfer.deepseek.moe as MO
orig = MO.MoE._forward_gpu_only
def spy(self, x, xf, indices, weights):
    lid = self.layer_id
    cap = torch.cuda.is_current_stream_capturing()
    r = orig(self, x, xf, indices, weights)
    if not cap:
        R.setdefault(lid, []).append(indices.view(-1).tolist())
    return r
MO.MoE._forward_gpu_only = spy

with torch.inference_mode():
    e.forward(ids, 0); e.forward(ids, 0); torch.cuda.synchronize()
    e.pin_hot_experts(); torch.cuda.synchronize()
    R.clear()
    e.reset(); og = e._forward_impl(ids, 1); torch.cuda.synchronize()
    print("EAGER1-ROUTES L0=%s L1=%s" % (R.get(0), R.get(1)), flush=True)
    R.clear()
    e.reset(); og2 = e.forward(ids, 1); torch.cuda.synchronize()
    print("CAPTURE-OUT %.4g" % float(e._graph["out"].abs().max()), flush=True)
    print("ARGMASK graph=%d eager=%d" % (int(e._graph["out"].argmax()), int(og.argmax())), flush=True)
    d = float((e._graph["out"].float() - og.float()).abs().max())
    print("DIFF %.4f" % d, flush=True)
