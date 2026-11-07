import sys, cProfile, pstats, io, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=[0, 1])
b = eng.block(0)
x = torch.randn(1, 1, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
with torch.inference_mode():
    for _ in range(3): b.attn.forward(x, 5)
    torch.cuda.synchronize()
    pr = cProfile.Profile()
    pr.enable()
    for p in range(6, 16):
        b.attn.forward(x, p)
    torch.cuda.synchronize()
    pr.disable()
    st = io.StringIO()
    pstats.Stats(pr, stream=st).sort_stats("cumulative").print_stats(18)
    print(st.getvalue()[:4000])
