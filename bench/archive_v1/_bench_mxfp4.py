
import torch
import vllm._C_stable_libtorch  # noqa
import vllm._moe_C_stable_libtorch  # noqa

dev = torch.device("cuda")
FP4_ID = 562949953487106

def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0

def get_scale_perms():
    sp, sps = [], []
    for i in range(8):
        sp.extend([i + 8 * j for j in range(8)])
    for i in range(4):
        sps.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return sp, sps

def marlin_permute_scales(s, size_k, size_n, group_size):
    sp, sps = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(sp)))[:, sp]
    else:
        s = s.reshape((-1, len(sps)))[:, sps]
    return s.reshape((-1, size_n)).contiguous()

def mxfp4_marlin_process_scales(ms):
    ms = ms.view(-1, 4)[:, [0, 2, 1, 3]].view(ms.size(0), -1)
    return ms.to(torch.float8_e8m0fnu)

N, K = 16384, 5120
# mxfp4 checkpoint style: fp4 [N, K/2] uint8, e8m0 scale [N, K/32]
w_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
w_scale = torch.randint(120, 128, (N, K // 32), dtype=torch.uint8, device=dev)

qw = torch.ops._C.gptq_marlin_repack(w_fp4.view(torch.int32).T.contiguous(), K, N, 4, False)
sc = marlin_permute_scales(w_scale.T.view(torch.float8_e8m0fnu), K, N, 32)
sc = mxfp4_marlin_process_scales(sc)
nbytes = qw.numel() * qw.element_size()
print("qweight", tuple(qw.shape), "bytes", round(nbytes/1e6, 1), "MB")
print("scales", tuple(sc.shape), sc.dtype)
ws = torch.zeros(26, dtype=torch.int32, device=dev)

for M in (1, 8, 16, 32, 64):
    a = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    out = torch.ops._C.marlin_gemm(a, None, qw, None, sc, None, None, None, ws, FP4_ID, M, N, K, False, False, False)
    t = bench(lambda: torch.ops._C.marlin_gemm(
        a, None, qw, None, sc, None, None, None, ws, FP4_ID, M, N, K, False, False, False))
    print(f"mxfp4 marlin M={M:3d}: {t:8.1f} us   {nbytes/(t*1e-6)/1e9:6.1f} GB/s")
