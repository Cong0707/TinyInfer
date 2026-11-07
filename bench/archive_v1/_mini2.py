
import torch, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
import tinyinfer.deepseek.model as M
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0, 1])
hc = eng.block(0).hc_mult

torch.manual_seed(0)
x0 = torch.randn(1, 1, hc, 5120, dtype=torch.bfloat16, device="cuda") * 0.1
pm0 = torch.zeros(1, 1, hc, dtype=torch.float32, device="cuda"); pm0[:, :, 0] = 1.0

res = {}
with torch.inference_mode():
    for mode in (False, True):
        M._FUSE_HC_PRE = mode
        h, pm = x0, pm0
        chain = []
        for lid in (0, 1):
            blk = eng.block(lid)
            h, pm = blk.forward(h, 0, pm)
            torch.cuda.synchronize()
            chain.append(h.clone())
        res[mode] = chain

for i, (a, b) in enumerate(zip(res[False], res[True])):
    diff = (a.float() - b.float()).abs().max().item()
    print(f"after block {i}: shape={tuple(a.shape)} maxdiff={diff:.3e} bit={bool((a==b).all())}")
