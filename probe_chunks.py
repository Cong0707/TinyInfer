"""分 chunk 处理 2048：第几个 chunk 开始坏。"""
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
b = e.block(0)
chunk = e.prefill_chunk
print("prefill_chunk=%d  N=%d -> %d chunks" % (chunk, N, (N+chunk-1)//chunk), flush=True)
e.reset()

for c0 in range(0, N, chunk):
    seg = ids[:, c0:c0+chunk]
    try:
        lg = e._forward_chunk(seg, c0)
        torch.cuda.synchronize()
        f = lg.float()
        ok = bool(torch.isfinite(f).all())
        if not ok or c0 < 3 * chunk or c0 > N - 3*chunk:
            print("  chunk@%4d len=%d finite=%-5s absmax=%s" % (
                  c0, seg.shape[1], ok,
                  ("%.3f" % float(f.abs().max())) if ok else "nan"), flush=True)
        if not ok:
            break
    except Exception as ex:
        print("  chunk@%4d EXC %s" % (c0, str(ex)[:70]), flush=True)
        break
