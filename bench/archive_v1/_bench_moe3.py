
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

TOPK, H = 8, 5120
print("== moe_sum  input [T, topk, H] -> [T, H] ==")
for T in (1, 8, 32, 128):
    inp = torch.randn(T, TOPK, H, dtype=torch.bfloat16, device=dev)
    outT = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    try:
        t = bench(lambda: torch.ops._moe_C.moe_sum(inp, outT, None, None))
        print(f"  T={T:4d}: {t:8.2f} us   {T*TOPK*H*2*2/(t*1e-6)/1e9:6.1f} GB/s")
    except Exception as ex:
        print("  T=", T, "FAIL", str(ex)[:140]); break

print("== moe_permute / moe_unpermute (E=256, topk=8) ==")
E = 256
for T in (1, 8, 32):
    topk_ids = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device=dev)
    tei = torch.arange(T * TOPK, dtype=torch.int32, device=dev).view(T, TOPK)
    inp = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    maxp = T * TOPK + E * 128
    pi = torch.empty(maxp, H, dtype=torch.bfloat16, device=dev)
    efto = torch.empty(E + 1, dtype=torch.int32, device=dev)
    inv = torch.empty(maxp, dtype=torch.int32, device=dev)
    pidx = torch.empty(maxp, dtype=torch.int32, device=dev)
    try:
        t = bench(lambda: torch.ops._moe_C.moe_permute(
            inp, topk_ids, tei, None, E, E, TOPK, pi, efto, inv, pidx), warmup=5, iters=50)
        print(f"  permute T={T:4d}: {t:8.2f} us")
    except Exception as ex:
        print("  permute T=", T, "FAIL", str(ex)[:140]); break
