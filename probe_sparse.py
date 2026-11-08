import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kernels.sparse_attn import sparse_attn_cuda, sparse_attn_dispatch
from tinyinfer.nn import sparse_attn_ref

torch.manual_seed(0)
b1, b2, s, h, d, n, k = 1, 2, 1, 8, 64, 128, 16
q1 = torch.randn(b1, s, h, d, device="cuda", dtype=torch.bfloat16) * 0.1
q2 = torch.cat([q1, torch.randn(b2 - b1, s, h, d, device="cuda", dtype=torch.bfloat16) * 0.1], 0)
kv1 = torch.randn(b1, n, d, device="cuda", dtype=torch.bfloat16) * 0.1
kv2 = torch.cat([kv1, torch.randn(b2 - b1, n, d, device="cuda", dtype=torch.bfloat16) * 0.1], 0)
sink = torch.randn(h, device="cuda")
idx1 = torch.randint(0, n, (b1, s, k), device="cuda", dtype=torch.int32)
idx2 = torch.cat([idx1, torch.randint(0, n, (b2 - b1, s, k), device="cuda", dtype=torch.int32)], 0)

o1 = sparse_attn_dispatch(q1, kv1, sink, idx1, 0.1)
o2 = sparse_attn_dispatch(q2, kv2, sink, idx2, 0.1)
print("kernel row0 maxdiff = %.6f" % float((o2[0:1].float() - o1.float()).abs().max()), flush=True)

r1 = sparse_attn_ref(q1, kv1, sink, idx1, 0.1)
r2 = sparse_attn_ref(q2, kv2, sink, idx2, 0.1)
print("ref    row0 maxdiff = %.6f" % float((r2[0:1].float() - r1.float()).abs().max()), flush=True)
print("kernel vs ref (b=1) = %.6f" % float((o1.float() - r1.float()).abs().max()), flush=True)
print("kernel vs ref (b=2) = %.6f" % float((o2.float() - r2.float()).abs().max()), flush=True)
