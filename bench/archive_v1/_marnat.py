
import time, torch
import tinyinfer._C_stable_libtorch  # noqa  (对照用)
from tinyinfer import native
from tinyinfer import gemv_fp4_marlin as GM

dev = "cuda"
N, K = 16384, 5120
w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device=dev)
s = torch.full((N, K//32), 127, dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)

wgt = GM._prepared(w, s)   # repack 一次
ref = torch.ops._C.marlin_gemm(x, None, wgt.qweight, None, wgt.scales, None, None, None,
                               GM._workspace(x.device), GM.FP4_TYPE_ID,
                               1, N, K, False, False, False)
torch.cuda.synchronize()

MANGLED = "_ZN6marlin6MarlinILx1125899906909960ELx562949953487106ELx1125899906909960ELx2814749767106568ELi256ELi1ELi8ELi8ELb1ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi"
sms = torch.cuda.get_device_properties(0).multi_processor_count
smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
native.set_dynamic_smem("marlin_fp4", MANGLED, smem)
# A B C C_tmp bias a_s b_s g_s zp | m n k lda | locks | has_bias atomic fp32 smem
L = native.bind("marlin_fp4", MANGLED, (sms,), (256,),
                "TTOnnnTnni" + "iii" + "Ti" + "iii", shared=smem)
out = torch.empty(1, N, dtype=torch.bfloat16, device=dev)
L(x, wgt.qweight, out, None, None, None, wgt.scales, None, None,
  1, N, K, K, GM._workspace(x.device), 0, 0, 0, smem)
torch.cuda.synchronize()
print("max diff vs pyd:", float((ref - out).abs().max()))
print("rel:", float((ref - out).abs().max() / ref.abs().max()))

def bench(fn, R=100, w_=30):
    for _ in range(w_): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

u_ref = bench(lambda: torch.ops._C.marlin_gemm(x, None, wgt.qweight, None, wgt.scales, None, None, None,
             GM._workspace(x.device), GM.FP4_TYPE_ID, 1, N, K, False, False, False))
u_nat = bench(lambda: L(x, wgt.qweight, out, None, None, None, wgt.scales, None, None,
             1, N, K, K, GM._workspace(x.device), 0, 0, 0, smem))
mb = N*K/2/1e6
print(f"pyd : {u_ref:.1f} us  {mb/u_ref*1e3:.0f} GB/s")
print(f"ours: {u_nat:.1f} us  {mb/u_nat*1e3:.0f} GB/s  ({u_ref/u_nat:.2f}x)")
