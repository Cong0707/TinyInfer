
import torch
import tinyinfer._C_stable_libtorch  # noqa

E, N, K = 4, 1024, 512
rows_per = 8
M = E * rows_per

a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
b = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda")
# e8m0 scale 127 == 1.0
a_bs = torch.full((E, M, K // 32), 127, dtype=torch.uint8, device="cuda")
b_bs = torch.full((E, N, K // 32), 127, dtype=torch.uint8, device="cuda")
problem_sizes = torch.tensor([[rows_per, N, K]] * E, dtype=torch.int32, device="cuda")
expert_offsets = torch.arange(0, M + 1, rows_per, dtype=torch.int32, device="cuda")
sf_offsets = (torch.arange(E, dtype=torch.int32, device="cuda") * rows_per)
out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

print("shapes:", a.shape, b.shape, a_bs.shape, b_bs.shape)
print("problem_sizes:", problem_sizes.tolist(), "expert_offsets:", expert_offsets.tolist(), "sf_offsets:", sf_offsets.tolist())
try:
    torch.ops._C.cutlass_mxfp4_group_mm(out, a, b, a_bs, b_bs, problem_sizes, expert_offsets, sf_offsets)
    torch.cuda.synchronize()
    print("RUN OK, out sample:", out[0, :4].tolist())
    print("nonzero rows:", int((out.abs().sum(dim=1) > 0).sum()))
except Exception as exc:
    print("FAILED:", type(exc).__name__, str(exc)[:400])
