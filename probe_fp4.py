"""reset 后 fp4 权重是否含 NaN；以及 MoE 用到的槽位字节是否正确。"""
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
sl = st.slots

def probe(tag):
    rec = st.fetch(0, [3], cache=True)[0]
    ex = moe._make_expert(rec)
    fp4 = ex.fp4
    parts = []
    for i, t_ in enumerate(fp4):
        if torch.is_tensor(t_):
            if t_.dtype == torch.uint8:
                parts.append("t%d=u8%s" % (i, tuple(t_.shape)[:2]))
            else:
                f = t_.float()
                parts.append("t%d=%s finite=%s absmax=%.3f" % (
                             i, str(t_.dtype).replace("torch.",""),
                             bool(torch.isfinite(f).all()), float(f.abs().max())))
    print("%-16s %s" % (tag, " ".join(parts)), flush=True)

e.reset()
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
probe("after prefill")
e.reset()
probe("after reset")
