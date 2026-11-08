import sys, torch
sys.path.insert(0, "/models/tinyinfer/src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = "/models/deepseek"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
pad = tok.encode("你好", add_special_tokens=False)[0]
ids = torch.tensor([[pad]], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1], tokenizer=tok)

def snap(tag):
    sl = e.expert_store.slots
    out = {"tag": tag}
    a = getattr(sl, "_arena", None)
    out["arena_ptr"] = int(a[0].data_ptr()) if a else None
    out["arena_shape"] = tuple(a[0].shape) if a else None
    t = getattr(sl, "_slot_tab", None)
    out["tab_ptr"] = int(t.data_ptr()) if t is not None else None
    for lid in (0, 1):
        ffn = e.block(lid).ffn
        for nm in ("_scratch_buf", "_gpu_y_buf", "_sids_buf", "_aq_buf", "_asc_buf", "_gpu_o_buf"):
            v = getattr(ffn, nm, None)
            out["L%d.%s" % (lid, nm)] = int(v.data_ptr()) if torch.is_tensor(v) else None
        at = e.block(lid).attn
        st = at._st()
        for nm in ("winbuf", "kv"):
            v = st.get(nm)
            out["L%d.attn.%s" % (lid, nm)] = int(v.data_ptr()) if torch.is_tensor(v) else None
    return out

with torch.inference_mode():
    e.forward(ids, 0); e.forward(ids, 0); torch.cuda.synchronize()
    e.pin_hot_experts(); torch.cuda.synchronize()
    before = snap("BEFORE")
    e.reset(); e.forward(ids, 1); torch.cuda.synchronize()
    after = snap("AFTER")
    print("GRAPH %.4g" % float(e._graph["out"].abs().max()), flush=True)

for k in before:
    if k == "tag":
        continue
    b, a = before[k], after[k]
    if b != a:
        print("CHANGED %-24s %s -> %s" % (k, b, a), flush=True)
print("DUMP-DONE", flush=True)
