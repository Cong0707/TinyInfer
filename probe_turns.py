"""能工作的配置（4 层）下，多轮 reset 是否稳定。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
qs = ["1+1 等于几？只回答数字。", "2+2 等于几？只回答数字。", "3+3 等于几？只回答数字。"]

for nlayers in (4, 5):
    try:
        e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda",
                   layers=list(range(nlayers)), tokenizer=tok)
    except Exception as ex:
        print("n=%d engine EXC %s" % (nlayers, str(ex)[:50]), flush=True)
        break
    for i, q in enumerate(qs):
        e.reset()
        ids = torch.tensor([tok.encode(q, add_special_tokens=False)], device="cuda")
        lg = e._forward_chunk(ids, 0)
        torch.cuda.synchronize()
        f = lg.float()
        ok = bool(torch.isfinite(f).all())
        nxt = int(f[0, -1].argmax()) if ok else -1
        print("n=%d turn%d finite=%-5s argmax=%d" % (nlayers, i, ok, nxt), flush=True)
    del e
    import gc; gc.collect(); torch.cuda.empty_cache()
