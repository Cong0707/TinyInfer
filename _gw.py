import sys, torch
sys.path.insert(0, "/models/tinyinfer/src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = "/models/deepseek"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1], tokenizer=tok)
for lid in (0, 1):
    g = e.block(lid).ffn.gate
    print("L%d gate type=%s" % (lid, type(g).__name__), flush=True)
    for nm in ("weight", "w", "bias", "correction_bias", "e_score_correction_bias"):
        v = getattr(g, nm, None)
        if torch.is_tensor(v):
            print("   %s device=%s dtype=%s shape=%s" % (nm, v.device, v.dtype, tuple(v.shape)), flush=True)
    for nm, v in vars(g).items():
        if torch.is_tensor(v) and v.device.type == "cpu":
            print("   CPU-TENSOR %s shape=%s" % (nm, tuple(v.shape)), flush=True)
