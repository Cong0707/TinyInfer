
import torch, time
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import native
dev="cuda"; D=4096
x = torch.randn(1, D, dtype=torch.bfloat16, device=dev)
w = torch.randn(D, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(x)

def gpu_time(fn, R=2000):
    for _ in range(200): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(R): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/R*1000

# 扫 block size，看哪个接近 vLLM
for blk in (128, 256, 512, 1024):
    L = native.bind("vllm_rms_norm","vllm_rms_norm_bf16",(1,),(blk,),"TTTif")
    t = gpu_time(lambda: L(out, x, w, D, 1e-6))
    print(f"block={blk:5}  {t:5.2f} us")
print("vLLM _C    :", f"{gpu_time(lambda: torch.ops._C.rms_norm(out,x,w,1e-6)):5.2f} us")
