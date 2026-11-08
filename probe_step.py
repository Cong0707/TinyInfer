"""手动 chunk=32 且带 reset_prefill_pool，复现 cold NaN。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
base = tok.encode("你好今天天气不错", add_special_tokens=False)
N = 2048
ids = torch.tensor([(base*2000)[:N]], device="cuda")
STEP = int(sys.argv[1])
DORESET = sys.argv[2] == "1"
print("STEP=%d reset=%s" % (STEP, DORESET), flush=True)
e.reset()
store = getattr(e, "expert_store", None)
bad = -1
for off in range(0, N, STEP):
    lg = e._forward_chunk(ids[:, off:off+STEP], off)
    if DORESET and store is not None:
        r = getattr(store, "reset_prefill_pool", None)
        if r is not None:
            r()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(lg).all()):
        bad = off
        break
print("RESULT STEP=%d reset=%s bad_chunk=%d" % (STEP, DORESET, bad), flush=True)
