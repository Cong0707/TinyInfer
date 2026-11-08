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
with torch.inference_mode():
    e.forward(ids, 0); e.forward(ids, 0); torch.cuda.synchronize()
    e.pin_hot_experts(); torch.cuda.synchronize()
    e.reset(); oe = e._forward_impl(ids, 1); torch.cuda.synchronize()
    print("EAGER %.4g" % float(oe.abs().max()), flush=True)

    # 第一次捕获
    e.reset(); e.forward(ids, 1); torch.cuda.synchronize()
    g1 = e._graph
    print("CAP1  %.4g  ptr=%d" % (float(g1["out"].abs().max()), g1["out"].data_ptr()), flush=True)

    # 强制第二次捕获：清掉 _graph 并让它重建
    e._graph = None
    e._graph_needed = True
    e.reset(); e.forward(ids, 1); torch.cuda.synchronize()
    g2 = e._graph
    print("CAP2  %.4g  ptr=%d" % (float(g2["out"].abs().max()), g2["out"].data_ptr()), flush=True)

    # 第三次
    e._graph = None
    e._graph_needed = True
    e.reset(); e.forward(ids, 1); torch.cuda.synchronize()
    g3 = e._graph
    print("CAP3  %.4g  ptr=%d" % (float(g3["out"].abs().max()), g3["out"].data_ptr()), flush=True)
