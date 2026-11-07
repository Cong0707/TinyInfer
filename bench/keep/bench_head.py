
import time, os, sys
import torch, psutil

torch.set_num_threads(16)
print("CPU threads", torch.get_num_threads(), "RAM avail GB", round(psutil.virtual_memory().available/2**30,2))
V, D = 129280, 5120
g = torch.Generator().manual_seed(0)

# CPU bf16 head (the checkpoint dtype)
w_bf16 = torch.randn(V, D, dtype=torch.bfloat16, generator=g)
h_bf16 = torch.randn(D, dtype=torch.bfloat16, generator=g)
for _ in range(2): y = torch.mv(w_bf16, h_bf16)
t = time.perf_counter(); n = 5
for _ in range(n): y = torch.mv(w_bf16, h_bf16)
dt = (time.perf_counter()-t)/n
print(f"CPU bf16 mv head: {dt*1e3:.1f} ms  ({V*D*2/dt/1e9:.1f} GB/s effective)")
print("  argmax", int(y.argmax()))

# CPU fp32 (what model.py does today: .float())
w_f32 = w_bf16.float(); h_f32 = h_bf16.float()
for _ in range(2): y = torch.mv(w_f32, h_f32)
t = time.perf_counter(); n = 3
for _ in range(n): y = torch.mv(w_f32, h_f32)
dt = (time.perf_counter()-t)/n
print(f"CPU fp32 mv head: {dt*1e3:.1f} ms (plus 2.65GB conversion cost)")
del w_f32, h_f32

# GPU bf16 GEMV + H2D
if torch.cuda.is_available():
    torch.cuda.init()
    print("VRAM used after init MB", round(torch.cuda.memory_allocated()/2**20,1))
    w_gpu = w_bf16.cuda()
    h_gpu = h_bf16.cuda()
    for _ in range(5): y = torch.mv(w_gpu, h_gpu)
    torch.cuda.synchronize(); t=time.perf_counter(); n=50
    for _ in range(n): y = torch.mv(w_gpu, h_gpu)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/n
    print(f"GPU bf16 mv head (resident): {dt*1e3:.3f} ms")
    del w_gpu, h_gpu
    torch.cuda.empty_cache()
    # H2D 1.324GB pageable vs pinned
    for pin in (False, True):
        src = torch.empty(V, D, dtype=torch.bfloat16, pin_memory=pin)
        dst = torch.empty_like(src, device="cuda")
        for _ in range(1): dst.copy_(src)
        torch.cuda.synchronize(); t=time.perf_counter(); n=3
        for _ in range(n): dst.copy_(src)
        torch.cuda.synchronize(); dt=(time.perf_counter()-t)/n
        print(f"H2D bf16 head pinned={pin}: {dt*1e3:.1f} ms ({V*D*2/dt/1e9:.2f} GB/s)")
        del src, dst
    torch.cuda.empty_cache()

    # fp8 head: per-row scale, dequant on GPU then GEMV; also H2D
    w8 = (w_bf16.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
    wq = (w_bf16.float() / w8).clamp(-448, 448).to(torch.float8_e4m3fn)
    wq = wq.contiguous()
    src8 = torch.empty(V, D, dtype=torch.float8_e4m3fn, pin_memory=True)
    src8.copy_(wq)
    dst8 = torch.empty_like(wq, device="cuda")
    torch.cuda.synchronize(); t=time.perf_counter(); n=3
    for _ in range(n): dst8.copy_(src8)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/n
    print(f"H2D fp8 head 0.66GB pinned: {dt*1e3:.1f} ms ({V*D/dt/1e9:.2f} GB/s)")
    s_gpu = w8.cuda()
    hh = h_bf16.cuda()
    # dequant cost + gemv
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(10):
        wd = (dst8.to(torch.bfloat16) * s_gpu.to(torch.bfloat16))
        y = torch.mv(wd, hh)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/10
    print(f"GPU fp8->bf16 dequant+mv: {dt*1e3:.2f} ms")
    # try _scaled_mm with per-row scale (A [1,D] bf16 -> cast fp8? simpler: 1xK fp8)
    try:
        a = (h_bf16 / (h_bf16.abs().max()/448)).to(torch.float8_e4m3fn).view(1, D).cuda()
        b = dst8.t()  # [D, V] fp8, needs column-major? torch requires b as [K,N]
        sb = w8.view(1, V).to(torch.float32).cuda()
        sa = torch.ones(1,1, dtype=torch.float32, device="cuda")
        out = torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        print("torch._scaled_mm fp8 head OK, out", tuple(out.shape))
    except Exception as e:
        print("torch._scaled_mm fp8 head FAILED:", type(e).__name__, str(e)[:180])
    del dst8, src8, s_gpu, hh
    torch.cuda.empty_cache()
    # int8 head dequant path size/timing (per-row scale)
    wq8 = (w_bf16.float() / w8).clamp(-127,127).to(torch.int8).contiguous()
    src_i8 = torch.empty(V, D, dtype=torch.int8, pin_memory=True); src_i8.copy_(wq8)
    dst_i8 = torch.empty_like(wq8, device="cuda")
    torch.cuda.synchronize(); t=time.perf_counter(); n=3
    for _ in range(n): dst_i8.copy_(src_i8)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/n
    print(f"H2D int8 head 0.66GB pinned: {dt*1e3:.1f} ms ({V*D/dt/1e9:.2f} GB/s)")
