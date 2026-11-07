
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

E, TOPK, BS = 256, 8, 128

print("== moe_align_block_size (E=256, topk=8, block=128) ==")
for T in (1, 8, 32, 128):
    topk_ids = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device=dev)
    maxpad = T * TOPK + E * BS
    sid = torch.empty(maxpad, dtype=torch.int32, device=dev)
    eid = torch.empty(E + 1, dtype=torch.int32, device=dev)
    ntp = torch.empty(1, dtype=torch.int32, device=dev)
    try:
        t = bench(lambda: torch.ops._moe_C.moe_align_block_size(topk_ids, E, BS, sid, eid, ntp, None, None))
        print(f"  T={T:4d}: {t:8.2f} us")
    except Exception as ex:
        print("  T=", T, "FAIL", str(ex)[:120]); break

print("== moe_sum (E=256, topk=8, hidden=5120) ==")
for T in (1, 8, 32):
    inp = torch.randn(T * TOPK, 5120, dtype=torch.bfloat16, device=dev)
    outT = torch.empty(T, 5120, dtype=torch.bfloat16, device=dev)
    try:
        t = bench(lambda: torch.ops._moe_C.moe_sum(inp, outT, None, None))
        print(f"  T={T:4d}: {t:8.2f} us   {T*TOPK*5120*2/(t*1e-6)/1e9:6.1f} GB/s")
    except Exception as ex:
        print("  T=", T, "FAIL", str(ex)[:150]); break

print("== gptq_gemm (int4, N=16384 K=5120) ==")
try:
    N, K = 16384, 5120
    a = torch.randn(1, K, dtype=torch.bfloat16, device=dev).to(torch.half)
    qw = torch.randint(0, 2**31, (K // 8, N), dtype=torch.int32, device=dev)
    qz = torch.randint(0, 2**31, (K // 8, N), dtype=torch.int32, device=dev)
    sc = torch.randn(K // 128, N, dtype=torch.half, device=dev)
    t = bench(lambda: torch.ops._C.gptq_gemm(a, qw, qz, sc, False, False, 4), warmup=3, iters=20)
    print(f"  M=1: {t:8.2f} us")
except Exception as ex:
    print("  FAIL", str(ex)[:200])

print("== awq_gemm (int4, N=16384 K=5120) ==")
try:
    N, K = 16384, 5120
    a = torch.randn(1, K, dtype=torch.half, device=dev)
    k = torch.randint(0, 2**31, (K // 8, N), dtype=torch.int32, device=dev)
    sf = torch.randn(K // 128, N, dtype=torch.half, device=dev)
    z = torch.randint(0, 2**31, (K // 128, N // 8), dtype=torch.int32, device=dev)
    t = bench(lambda: torch.ops._C.awq_gemm(a, k, sf, z, 1), warmup=3, iters=20)
    print(f"  M=1: {t:8.2f} us")
except Exception as ex:
    print("  FAIL", str(ex)[:200])
