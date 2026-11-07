import sys, cProfile, pstats, io, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=[0,1,2,3,4])
eng.enable_prefetch = False
prompt = list(range(16))
with torch.inference_mode():
    logits = eng.forward(torch.tensor([prompt]), 0)
    nid = int(logits[0, -1].argmax())
    eng.forward(torch.tensor([[nid]]), 16)
    torch.cuda.synchronize()
    pr = cProfile.Profile()
    pr.enable()
    for p in range(17, 27):
        eng.forward(torch.tensor([[nid]]), p)
    torch.cuda.synchronize()
    pr.disable()
    st = io.StringIO()
    pstats.Stats(pr, stream=st).sort_stats("tottime").print_stats(14)
    print(st.getvalue()[:3200])
