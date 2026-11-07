
import os
os.environ["TI_FUSE"] = "1"
import torch, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
import tinyinfer.deepseek.model as M
import tinyinfer.deepseek.hc as H
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
hc = eng.block(0).hc_mult
torch.manual_seed(0)
x0 = torch.randn(1, 1, hc, 5120, dtype=torch.bfloat16, device="cuda") * 0.1
pm0 = torch.zeros(1, 1, hc, dtype=torch.float32, device="cuda"); pm0[:, :, 0] = 1.0

chains = {}
with torch.inference_mode():
    for use in (False, True):
        M._FUSE_HC_PRE = use
        h, pm = x0, pm0
        chain = []
        for lid in range(12):
            h, pm = eng.block(lid).forward(h, 0, pm)
            torch.cuda.synchronize()
            chain.append(h.clone())
        chains[use] = chain
for lid in range(12):
    a = chains[False][lid]; b = chains[True][lid]
    d_ = (a.float()-b.float()).abs().max().item()
    tag = "resident" if lid < eng.gpu_layers else "STREAM"
    print(f"layer {lid:2d} ({tag:8s}) maxdiff={d_:.3e} bit={bool((a==b).all())}")
