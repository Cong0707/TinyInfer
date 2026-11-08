"""reset 之后专家权重是否含 NaN。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
moe = e.block(0).ffn
st = moe._expert_store

e.reset()
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
sl = st.slots
print("after prefill: slots occupied=%d" % len(sl._by_key), flush=True)

e.reset()
print("after reset:   slots occupied=%d" % len(sl._by_key), flush=True)

# 取几个专家看权重
for eid in (11, 25, 3):
    try:
        rec = st.fetch(0, [eid], cache=True)[0]
        ex = moe._make_expert(rec)
        w = ex.w1.weight if hasattr(ex, "w1") else None
        if w is not None:
            wf = w.float() if w.dtype != torch.uint8 else None
            if wf is not None:
                print("  expert %3d: shape=%s finite=%s absmax=%.4f" % (
                      eid, tuple(w.shape), bool(torch.isfinite(wf).all()),
                      float(wf.abs().max())), flush=True)
            else:
                print("  expert %3d: dtype=%s (fp4, skip)" % (eid, w.dtype), flush=True)
    except Exception as ex:
        print("  expert %3d EXC %s" % (eid, str(ex)[:60]), flush=True)
