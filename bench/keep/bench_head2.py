
import time, torch, psutil
V, D = 129280, 5120
g = torch.Generator().manual_seed(0)
w = torch.randn(V, D, dtype=torch.bfloat16, generator=g)
h = torch.randn(D, dtype=torch.bfloat16, generator=g)
print("RAM avail GB", round(psutil.virtual_memory().available/2**30,2))
for nt in (8, 16, 32):
    torch.set_num_threads(nt)
    y = torch.mv(w, h)
    t = time.perf_counter(); n = 5
    for _ in range(n): y = torch.mv(w, h)
    dt = (time.perf_counter()-t)/n
    print(f"CPU threads={nt:2d} bf16 mv: {dt*1e3:6.2f} ms ({V*D*2/dt/1e9:.1f} GB/s)")
# chunked matvec (better cache behavior?)
torch.set_num_threads(16)
for chunk in (4096, 8192, 16384):
    outs = []
    t = time.perf_counter(); n=3
    for _ in range(n):
        outs = [torch.mv(w[i:i+chunk], h) for i in range(0, V, chunk)]
    dt=(time.perf_counter()-t)/n
    print(f"CPU chunked {chunk}: {dt*1e3:.2f} ms")
# bf16 vs fp32 argmax agreement (random data; real test later)
torch.set_num_threads(16)
y_bf = torch.mv(w, h)
y_fp = torch.mv(w.float(), h.float())
print("argmax equal:", int(y_bf.argmax()) == int(y_fp.argmax()),
      "top5 overlap:", len(set(y_bf.topk(5).indices.tolist()) & set(y_fp.topk(5).indices.tolist())))
del y_fp
if torch.cuda.is_available():
    # per-tensor scale fp8 _scaled_mm on sm_120
    try:
        w8 = (w.float().abs().max()/448.0).clamp_min(1e-12)
        wq = (w.float()/w8).clamp(-448,448).to(torch.float8_e4m3fn).t().contiguous().cuda()  # [D,V]
        a = (h.float()/(h.float().abs().max()/448)).clamp(-448,448).to(torch.float8_e4m3fn).view(1,D).cuda()
        sa = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
        sb = torch.tensor([[float(w8)]], dtype=torch.float32, device="cuda")
        for _ in range(3): out = torch._scaled_mm(a, wq, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        torch.cuda.synchronize(); t=time.perf_counter(); n=20
        for _ in range(n): out = torch._scaled_mm(a, wq, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        torch.cuda.synchronize(); dt=(time.perf_counter()-t)/n
        print(f"GPU fp8 _scaled_mm head resident: {dt*1e3:.3f} ms  out={tuple(out.shape)}")
    except Exception as e:
        print("per-tensor fp8 _scaled_mm failed:", type(e).__name__, str(e)[:200])
