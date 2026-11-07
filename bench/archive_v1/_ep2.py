import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.ep import ExpertRoute, reduce_partials
# 模拟：2 rank 各算一半专家，结果应该等于单卡全算
print('RESULT === EP 数学验证 ===')
torch.manual_seed(0)
# 单卡：6 个专家的贡献全部累加
def moe_single(experts, x, w):
    y = torch.zeros_like(x)
    for e in experts:
        y += x * w[e]
    return y
# 双卡：rank0 算前 3 个，rank1 算后 3 个，然后求和
x = torch.randn(2, 8)
w = torch.randn(384)
experts = [3, 17, 42, 100, 201, 355]
route = ExpertRoute(384, ['cuda:0', 'cuda:1'])
full = moe_single(experts, x, w)
partials = []
for rank in (0, 1):
    mine = route.mine(experts, rank)
    partials.append(moe_single(mine, x, w))
ep = reduce_partials(partials)
print(f'  single: {full.flatten()[:4].tolist()}')
print(f'  ep    : {ep.flatten()[:4].tolist()}')
print(f'  maxdiff = {(full-ep).abs().max().item():.3e}')
print(f'  correct = {torch.allclose(full, ep)}')
# 反面验证：只取一个 rank 会错
only0 = partials[0]
print(f'  only rank0 maxdiff = {(full-only0).abs().max().item():.3e}  <- 这就是之前的 bug')