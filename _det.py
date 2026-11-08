import sys, torch
sys.path.insert(0, "/models/tinyinfer/src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = "/models/deepseek"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
pad = tok.encode("你好", add_special_tokens=False)[0]
ids = torch.tensor([[pad]], device="cuda")

for layers in ([0], [0, 1]):
    e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=layers, tokenizer=tok)
    with torch.inference_mode():
        e.forward(ids, 0); e.forward(ids, 0); torch.cuda.synchronize()
        e.pin_hot_experts(); torch.cuda.synchronize()
        # 只做 eager，不捕获：连跑两次，看 L0 输出是否稳定
        e.reset(); a = e._forward_impl(ids, 1); torch.cuda.synchronize()
        e.reset(); b = e._forward_impl(ids, 1); torch.cuda.synchronize()
        d = float((a.float() - b.float()).abs().max())
        print("LAYERS %s eager-vs-eager diff=%.6g  a=%.4g b=%.4g" % (
            layers, d, float(a.abs().max()), float(b.abs().max())), flush=True)
    del e
    import gc; gc.collect(); torch.cuda.empty_cache()
    break
