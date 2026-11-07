import sys, cProfile, pstats, io, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(19)), gpu_layers=19, head_on_cpu=True)
eng.enable_prefetch = False
prompt = list(range(8))
with torch.inference_mode():
    logits = eng.forward(torch.tensor([prompt]), 0)
    nid = int(logits[0, -1].argmax())
    for p in range(8, 12):
        eng.forward(torch.tensor([[nid]]), p)
    torch.cuda.synchronize()
    a = eng.block(2).attn
    x = torch.randn(1, 1, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
    for _ in range(3): a.forward(x, 6)
    torch.cuda.synchronize()
    pr = cProfile.Profile()
    pr.enable()
    for p in range(6, 14):
        a.forward(x, p)
    torch.cuda.synchronize()
    pr.disable()
    st = io.StringIO()
    pstats.Stats(pr, stream=st).sort_stats("tottime").print_stats(12)
    print(st.getvalue()[:2600])
