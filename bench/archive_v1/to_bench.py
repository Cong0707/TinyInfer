import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(19)), gpu_layers=8, head_on_cpu=True)
prompt = list(range(8))
with torch.inference_mode():
    logits = eng.forward(torch.tensor([prompt]), 0)
    nid = int(logits[0, -1].argmax())
    for p in range(8, 12):
        eng.forward(torch.tensor([[nid]]), p)
    torch.cuda.synchronize()
    b = eng.block(8)  # first streamed layer
    print("devices:", eng._layer_device(eng.block(7)), eng._layer_device(b))
    # to cuda
    t0 = time.perf_counter(); b.to("cuda"); torch.cuda.synchronize()
    print(f"block.to(cuda) first: {(time.perf_counter()-t0)*1e3:.1f} ms")
    b.to("cpu")
    t0 = time.perf_counter(); b.to("cuda"); torch.cuda.synchronize()
    print(f"block.to(cuda) cached: {(time.perf_counter()-t0)*1e3:.1f} ms")
    t0 = time.perf_counter(); b.to("cpu"); torch.cuda.synchronize()
    print(f"block.to(cpu): {(time.perf_counter()-t0)*1e3:.1f} ms")
    # weights only copy bandwidth test
    q = b.attn.wq_b if hasattr(b.attn, "wq_b") else None
    print("wq_b type:", type(q).__name__)
