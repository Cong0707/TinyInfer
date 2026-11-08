import sys, torch
sys.path.insert(0, "/models/tinyinfer/src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
n = int(sys.argv[1])
M = "/models/deepseek"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
pad = tok.encode("你好", add_special_tokens=False)[0]
ids = torch.tensor([[pad]], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(n)), tokenizer=tok)
with torch.inference_mode():
    e.forward(ids, 0); e.forward(ids, 0); torch.cuda.synchronize()
    e.pin_hot_experts(); torch.cuda.synchronize()
    e.reset(); oe = e._forward_impl(ids, 1); torch.cuda.synchronize()
    e.reset(); og = e.forward(ids, 1); torch.cuda.synchronize()
    g = e._graph
    d = float((g["out"].float()-oe.float()).abs().max())
    print("R n=%d graph=%.4g eager=%.4g diff=%.6g" % (n, float(g["out"].abs().max()),
          float(oe.abs().max()), d), flush=True)
