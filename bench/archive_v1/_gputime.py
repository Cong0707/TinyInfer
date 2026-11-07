
import time, torch
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import native
dev="cuda"; D=4096
x = torch.randn(1, D, dtype=torch.bfloat16, device=dev)
w = torch.randn(D, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(x)
launcher = native.bind("vllm_rms_norm","vllm_rms_norm_bf16",(1,),(512,),"TTTif")

# 用 CUDA event 测纯 GPU 时间（排除 CPU 启动开销）
def gpu_time(fn, R=2000):
    for _ in range(200): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(R): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/R*1000  # us

for _ in range(3):
    g_ours = gpu_time(lambda: launcher(out, x, w, D, 1e-6))
    g_ref  = gpu_time(lambda: torch.ops._C.rms_norm(out, x, w, 1e-6))
    print(f"GPU-only  ours={g_ours:5.2f}us  _C={g_ref:5.2f}us  ratio={g_ref/g_ours:.2f}x")
