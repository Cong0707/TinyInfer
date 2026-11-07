
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels import native
from tinyinfer.nn import sparse_attn_ref

torch.manual_seed(0)
B, M, H, D, N, K = 1, 1, 4, 64, 32, 16
q = torch.randn(B, M, H, D, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(B, N, D, dtype=torch.bfloat16, device="cuda")
sink = torch.randn(H, dtype=torch.float32, device="cuda")
idxs = torch.stack([torch.randperm(N)[:K] for _ in range(M)]).unsqueeze(0).int().cuda()
# 加一些 -1 填充
idxs[0, 0, -3:] = -1
scale = 1.0 / (D ** 0.5)

ref = sparse_attn_ref(q, kv, sink, idxs, scale)
out = torch.empty_like(q)
fn = native.kernel("sparse_attn_sink", "sparse_attn_sink")
blk = 128
smem = (blk // 32) * 4 + 32 * 4 + (blk // 32) * D * 4
native.launch(fn, (B * M, H), (blk,), [q, kv, sink, idxs, out, B, M, H, D, N, K, float(scale)], shared=smem)
torch.cuda.synchronize()
d = (out.float() - ref.float()).abs().max().item()
print(f"B{B} M{M} H{H} D{D} N{N} K{K}: maxdiff={d:.3e}  rel={d/max(1e-6, ref.float().abs().max().item()):.2e}")
