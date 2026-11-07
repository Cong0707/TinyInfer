import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(40)))
try:
    ok = eng.load_mtp()
    print("load_mtp:", ok, flush=True)
    if ok:
        for st in eng.mtp:
            print(f"  stage {st.stage}: attn={type(st.attn).__name__} block_size={st.block_size}", flush=True)
        # minimal forward: main prefill then stage chain
        with torch.inference_mode():
            logits = eng.forward(torch.tensor([[1, 2, 3, 4]]), 0)
            nid = int(logits[0, -1].argmax())
            hidden = eng._last_hidden  # [1,1,hc,dim]
            x, main_x = eng.mtp[0].forward_embed(hidden.reshape(1, -1) if hidden.dim() == 4 else hidden, torch.tensor([nid]))
            pre = torch.zeros(1, x.size(1), eng.cfg.hc_mult, device=x.device); pre[:,:,0] = 1.0
            for st in eng.mtp:
                x, pre = st.forward(x, 4, pre, main_x)
            print("mtp chain ok, x:", tuple(x.shape), flush=True)
except Exception:
    import traceback; traceback.print_exc()
