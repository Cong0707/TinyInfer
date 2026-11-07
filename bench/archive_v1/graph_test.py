import sys, time, torch
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
    for p in range(8, 14):
        eng.forward(torch.tensor([[nid]]), p)
    torch.cuda.synchronize()
    a = eng.block(0).attn
    x = torch.randn(1, 1, 5120, dtype=torch.bfloat16, device="cuda")
    # warmup on a private stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out = a.forward(x, 20)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    # capture
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            o_static = a.forward(x, 21)
        print("capture ok; replay timing:")
        def replay():
            g.replay()
        for _ in range(3): replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(20): replay()
        e1.record(); torch.cuda.synchronize()
        print(f"graph replay: {e0.elapsed_time(e1)/20:.2f} ms")
    except Exception as exc:
        print("capture failed:", type(exc).__name__, str(exc)[:300])
        # fallback timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20): a.forward(x, 22)
        torch.cuda.synchronize()
        print(f"eager: {(time.perf_counter()-t0)/20*1e3:.2f} ms")
