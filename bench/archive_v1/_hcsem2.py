
import torch
torch.manual_seed(0)
M, hc, D = 2, 3, 4
x    = torch.randn(M, D)
res  = torch.randn(M, hc, D)
post = torch.randn(M, hc)
comb = torch.randn(M, hc, hc)

ref = post.unsqueeze(-1)*x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1)*res.unsqueeze(-2), dim=2)

m, o, d = 0, 1, 2
tgt = ref[m, o, d].item()
base = (post[m,o]*x[m,d]).item()
print(f"ref[{m},{o},{d}] = {tgt:.6f}   base(post*x) = {base:.6f}   resid_part = {tgt-base:.6f}")
cands = {
 "sum_i comb[m,i,o]*res[m,i,d]": sum(comb[m,i,o].item()*res[m,i,d].item() for i in range(hc)),
 "sum_i comb[m,o,i]*res[m,i,d]": sum(comb[m,o,i].item()*res[m,i,d].item() for i in range(hc)),
 "sum_i comb[m,i,o]*res[m,o,d]": sum(comb[m,i,o].item() for i in range(hc))*res[m,o,d].item(),
 "res[m,o,d]*sum_j comb[m,o,j]": res[m,o,d].item()*sum(comb[m,o,j].item() for j in range(hc)),
 "sum_i comb[m,o,i]*res[m,o,d]": sum(comb[m,o,i].item() for i in range(hc))*res[m,o,d].item(),
}
for k,v in cands.items():
    print(f"  {k:<40} = {v:+.6f}   {'<<< MATCH' if abs(v-(tgt-base))<1e-5 else ''}")
