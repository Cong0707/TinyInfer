
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.nn import rms_norm
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
b0 = eng.block(0)
x1 = eng.embed[torch.tensor(p)].float() if hasattr(eng.embed, "__getitem__") else None
# 用 hc 前的输入直接测 attention
with torch.inference_mode():
    eng.reset()
    # 走引擎拿 layer0 的输入
    ids = torch.tensor([p], device="cuda")
    h1 = eng.embed[ids.cpu()].to(eng.device) if eng.embed_on_cpu else eng.embed[ids]
    h2 = h1.repeat(2, 1, 1)
    print("h1", tuple(h1.shape), "h2", tuple(h2.shape))
    pre1 = torch.zeros(1, h1.size(-1)*4, device=eng.device) if False else None
    # 用真实 pre_mix
    eng.reset(); out1 = b0.forward(h1, 0, None)
    eng.reset(); out2 = b0.forward(h2, 0, None)
    print("layer0 out1", tuple(out1[0].shape), "out2", tuple(out2[0].shape))
    dm = (out1[0].float() - out2[0][0:1].float()).abs().max().item()
    print("layer0 输出 maxdiff:", dm)
