
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
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single

def marlin_permute_scales(s, size_k, size_n, group_size):
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()

def nvfp4_marlin_process_scales(ms):
    ms = ms.to(torch.half)
    ms = ms.view(-1, 4)[:, [0, 2, 1, 3]].view(ms.size(0), -1)
    sf = 1.0
    wsf = ms.float() * (2**7)
    nz = wsf > 0
    if nz.any():
        mx = wsf[nz].max()
        if mx < 448 * (2**7):
            sf = (448 * (2**7) / mx).log2().floor().exp2().item()
    if sf > 1.0:
        ms = (ms.float() * sf).to(torch.half)
    ms = ms * (2**7)
    ms[ms < 2] = 0
    ms = ms.view(torch.int16) << 1
    ms = ms.view(torch.float8_e4m3fn)
    return ms[:, 1::2].contiguous(), sf

def make_fp4_nvfp4(weight, group_size=16):
    size_n, size_k = weight.shape
    scales = weight.view(size_n, -1, group_size).abs().max(-1)[0] / 6
    global_scale = scales.max() / 448
    scales = (scales / global_scale).to(torch.float8_e4m3fn)
    fp4_weight = torch.randint(0, 256, (size_n, size_k // 2), dtype=torch.uint8, device=weight.device)
    mq = torch.ops._C.gptq_marlin_repack(
        fp4_weight.view(torch.int32).T.contiguous(), size_k, size_n, 4, False)
    msc = marlin_permute_scales(scales.T.to(weight.dtype), size_k, size_n, group_size)
    msc, sf = nvfp4_marlin_process_scales(msc)
    gs = (global_scale * (2.0 ** (2**7 - 2**1 - 7))).to(torch.float32) / sf
    return mq, msc, gs

N, K = 16384, 5120
w = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
qw, sc, gs = make_fp4_nvfp4(w, 16)
nbytes = qw.numel() * qw.element_size()
print("qweight", tuple(qw.shape), "bytes", round(nbytes/1e6, 1), "MB")
ws = torch.zeros(26, dtype=torch.int32, device=dev)

def run(M, atomic=False, fp32red=False):
    a = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    return torch.ops._C.marlin_gemm(
        a, None, qw, None, sc, None, gs, None, ws, FP4_ID, M, N, K, atomic, fp32red, False)

out = run(1)
print("out", tuple(out.shape), out.dtype, "sample", out.flatten()[:3].tolist())

for M in (1, 8, 16, 32, 64, 128, 256):
    t = bench(lambda: run(M))
    print(f"marlin_gemm fp4 M={M:3d}: {t:8.1f} us   {nbytes/(t*1e-6)/1e9:6.1f} GB/s")
