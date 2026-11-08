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
sl = e.expert_store.slots

def report(tag):
    t = sl._slot_tab
    ptr = int(t.data_ptr())
    host = t[:2].cpu()
    nz0 = int((host[0] >= 0).sum()); nz1 = int((host[1] >= 0).sum())
    print("%-10s ptr=%d shape=%s nz=(%d,%d) by_key=%d" %
          (tag, ptr, tuple(t.shape), nz0, nz1, len(sl._by_key)), flush=True)
    return ptr

with torch.inference_mode():
    e.forward(ids, 0); e.forward(ids, 0); torch.cuda.synchronize()
    report("warmup")
    e.pin_hot_experts(); torch.cuda.synchronize()
    report("after-pin")
    # 捕获前再建一次（模拟 2826 的路径）
    r = sl.slot_table(int(e.max_layers), 384)
    sl.table_sync()
    torch.cuda.synchronize()
    report("table-call")
    e.reset(); e.forward(ids, 1); torch.cuda.synchronize()
    report("after-capture")
    print("GRAPH %.4g" % float(e._graph["out"].abs().max()), flush=True)
    g = e._graph
    print("GRAPH-PTR %d" % int(g["out"].data_ptr()), flush=True)
