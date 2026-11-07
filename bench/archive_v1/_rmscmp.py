
import time, torch, math
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import native

dev = "cuda"
def bench(fn, R=3000, w_=300):
    for _ in range(w_): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/R*1e6

kn = native.kernel("vllm_rms_norm", "vllm_rms_norm_bf16")

for (M, D) in [(1, 4096), (8, 4096), (1, 7168), (64, 5120)]:
    x = torch.randn(M, D, dtype=torch.bfloat16, device=dev)
    w = torch.randn(D, dtype=torch.bfloat16, device=dev)
    ref = torch.empty_like(x); mine = torch.empty_like(x)

    torch.ops._C.rms_norm(ref, x, w, 1e-6)
    vec = min(8, D)
    block = min(D // vec, 1024 if M < 256 else 256)
    def run_mine():
        native.launch(kn, (M,), (block,), (mine, x, w, D, 1e-6))
    run_mine(); torch.cuda.synchronize()
    diff = float((ref - mine).abs().max())
    rel = diff / float(ref.abs().max().clamp_min(1e-9))
    u_ref = bench(lambda: torch.ops._C.rms_norm(ref, x, w, 1e-6))
    u_mine = bench(run_mine)
    print(f"M={M:<3} D={D:<5} diff={diff:.5f} (rel {rel:.2e})  _C={u_ref:6.2f}us  ours={u_mine:6.2f}us  ratio={u_ref/u_mine:.2f}x")
