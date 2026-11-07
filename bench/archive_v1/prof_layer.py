import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", max_layers=1, device="cuda", tier=OptLevel.BALANCED)
b = eng.block(0)
x = torch.randn(1, 1, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
x_hc = x.unsqueeze(2).repeat(1, 1, 4, 1)
pre = torch.zeros(1, 1, 4, dtype=torch.float32, device="cuda"); pre[:, :, 0] = 1.0

def t(fn, n=5):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3

with torch.inference_mode():
    b.attn.reset(); b.ffn._expert_cache.clear()
    print("block.forward ms:", round(t(lambda: b.forward(x_hc, 5, pre)), 2))
    print("attn.forward  ms:", round(t(lambda: b.attn.forward(x, 5)), 2))
    print("wq_a          ms:", round(t(lambda: b.attn.wq_a.forward(x)), 2))
    print("wq_b          ms:", round(t(lambda: b.attn.wq_b.forward(torch.randn(1,1,1280,dtype=torch.bfloat16,device='cuda'))), 2))
    print("ffn.forward   ms:", round(t(lambda: b.ffn.forward(x)), 2))
    print("gate          ms:", round(t(lambda: b.ffn.gate.forward(x.view(-1,5120))), 2))
