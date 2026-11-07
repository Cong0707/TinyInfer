import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=[0,1,2,3,4,5,6])
prompt = list(range(16))
with torch.inference_mode():
    logits = eng.forward(torch.tensor([prompt]), 0)
    nid = int(logits[0, -1].argmax())
    eng.forward(torch.tensor([[nid]]), 16)  # warmup
    torch.cuda.synchronize()
    # timed token x3
    t0 = time.perf_counter()
    for p in (17, 18, 19):
        eng.forward(torch.tensor([[nid]]), p)
    torch.cuda.synchronize()
    print(f"decode total: {(time.perf_counter()-t0)/3*1e3:.1f} ms/token")
    # torch profiler on one token
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        eng.forward(torch.tensor([[nid]]), 20)
    torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14))
