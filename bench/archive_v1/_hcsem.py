
import torch
torch.manual_seed(0)
M, hc, D = 2, 3, 4
x    = torch.randn(M, D)
res  = torch.randn(M, hc, D)
post = torch.randn(M, hc)
comb = torch.randn(M, hc, hc)

a = post.unsqueeze(-1)      # ?
b = x.unsqueeze(-2)
print("post.unsqueeze(-1)", tuple(a.shape), " x.unsqueeze(-2)", tuple(b.shape))
part1 = a * b
print("part1", tuple(part1.shape))

c = comb.unsqueeze(-1)
d = res.unsqueeze(-2)
print("comb.unsqueeze(-1)", tuple(c.shape), " res.unsqueeze(-2)", tuple(d.shape))
prod = c * d
print("prod", tuple(prod.shape))
s = torch.sum(prod, dim=2)
print("sum(dim=2)", tuple(s.shape))
ref = part1 + s

# 手工两种写法
alt_A = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.einsum("mio,mod->mid", comb, res)
alt_B = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.einsum("moi,mod->mid", comb, res.transpose(1,2).transpose(1,2))
alt_B = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.einsum("moi,mad->mid", comb, res)
print("max|ref-A| (sum over source i):", float((ref-alt_A).abs().max()))
print("max|ref-B| (sum over resid  a):", float((ref-alt_B).abs().max()))
