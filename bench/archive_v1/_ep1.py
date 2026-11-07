"""Expert parallelism: does summing disjoint-rank partials equal the single-card result?"""
import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.ep import ExpertRoute, reduce_partials, load_balance

print('RESULT === ExpertRoute ===')
for world, caps in ((1, [1]), (2, [1, 1]), (4, [1, 1, 1, 1]), (2, [3, 1])):
    r = ExpertRoute(384, [f'cuda:{i}' for i in range(world)], caps)
    ok = all(r.rank_of(e) is not None for e in range(384))
    cover = sorted(e for e in range(384) if r.rank_of(e) >= 0)
    print(f'  world={world} caps={caps}: {r.summary()[:70]}')
    assert len(cover) == 384, 'every expert must have an owner'
    for rank in range(world):
        mine = r.mine(list(range(384)), rank)
        assert all(r.rank_of(e) == rank for e in mine)
    print(f'    OK: 384 experts covered, each owned exactly once')

print('RESULT === reduce_partials ===')
torch.manual_seed(0)
base = torch.randn(4, 8, device='cuda')
# 单卡：一次做完
p1 = torch.zeros(4, 8, device='cuda')
# 模拟两卡各算一半专家
a = torch.randn(4, 8, device='cuda')
b = torch.randn(4, 8, device='cuda')
want = a + b
got = reduce_partials([a, b])
print(f'  2 partials: maxdiff={(want-got).abs().max().item():.3e}')
got1 = reduce_partials([a])
print(f'  1 partial passthrough: {torch.equal(got1, a)}, same object={got1 is a}')
got4 = reduce_partials([a, b, torch.zeros_like(a), torch.zeros_like(a)])
print(f'  4 partials with zeros: maxdiff={(want-got4).abs().max().item():.3e}')

print('RESULT === load_balance ===')
r = ExpertRoute(384, ['cuda:0', 'cuda:1'])
import random
random.seed(0)
counts = [random.randint(1, 100) for _ in range(384)]
assign = load_balance(r, counts)
per_rank = [0, 0]
for e, rk in enumerate(assign):
    per_rank[rk] += counts[e]
orig = [sum(counts[lo:hi]) for (lo, hi) in r.block]
print(f'  original per-rank load: {orig}  imbalance={max(orig)/max(1,min(orig)):.2f}x')
print(f'  balanced per-rank load: {per_rank}  imbalance={max(per_rank)/max(1,min(per_rank)):.2f}x')
print(f'  every expert assigned: {len(assign) == 384}')