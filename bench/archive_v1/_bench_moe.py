
import torch
import vllm._C_stable_libtorch  # noqa
import vllm._moe_C_stable_libtorch  # noqa

dev = torch.device("cuda")
def bench(fn, warmup=10, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0

E, TOPK, NGROUP, TOPKG = 256, 8, 8, 4
print("== grouped_topk (E=256, topk=8, groups=8/4) ==")
for T in (1, 8, 32, 128, 512):
    scores = torch.randn(T, E, dtype=torch.float32, device=dev)
    bias = torch.zeros(E, dtype=torch.float32, device=dev)
    t = bench(lambda: torch.ops._moe_C.grouped_topk(scores, NGROUP, TOPKG, TOPK, True, 2.5, bias, 0))
    print(f"  T={T:4d}: {t:8.2f} us   {T*E*4/(t*1e-6)/1e9:6.1f} GB/s")

print("== topk_softmax (E=256, topk=8) ==")
for T in (1, 8, 32, 128, 512):
    gating = torch.randn(T, E, dtype=torch.float32, device=dev)
    w = torch.empty(T, TOPK, dtype=torch.float32, device=dev)
    idx = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
    tei = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
    t = bench(lambda: torch.ops._moe_C.topk_softmax(w, idx, tei, gating, True, None, None))
    print(f"  T={T:4d}: {t:8.2f} us   {T*E*4/(t*1e-6)/1e9:6.1f} GB/s")
