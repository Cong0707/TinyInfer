import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.ep import ExpertRoute, reduce_partials

# 用真实的专家权重，比较：单卡全算 vs 2-rank 分片后求和
from tinyinfer.weights import ShardStore
from tinyinfer.deepseek.fp4_native import gemv_fp4_pq, quantize_act
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
L = 3
experts = [3, 17, 42, 100, 201, 355]
k = 5120
x = torch.randn(1, k, dtype=torch.bfloat16, device='cuda')
aq, asc = quantize_act(x, k)
def run(experts, n_out):
    y = torch.zeros(n_out, dtype=torch.float32, device='cuda')
    for e in experts:
        w = st.get(f'layers.{L}.ffn.experts.{e}.w2.weight').to('cuda')
        s = st.get(f'layers.{L}.ffn.experts.{e}.w2.scale').to('cuda')
        y += gemv_fp4_pq(aq, asc, w, s).flatten().float()
    return y
route = ExpertRoute(384, ['cuda:0', 'cuda:1'])
# 单卡：全 6 个专家
full = run(experts, 5120)
# 双卡：各算自己那份
parts = []
for rank in (0, 1):
    mine = route.mine(experts, rank)
    print(f'RESULT rank{rank} owns {len(mine)} of {len(experts)}: {mine}')
    parts.append(run(mine, 5120))
ep = reduce_partials(parts)
print(f'RESULT full[:3] {full[:3].tolist()}')
print(f'RESULT ep  [:3] {ep[:3].tolist()}')
print(f'RESULT maxdiff {(full-ep).abs().max().item():.3e}  rel {(full-ep).abs().max().item()/max(1e-6,full.abs().max().item()):.2e}')
print(f'RESULT only rank0 would differ by {(full-parts[0]).abs().max().item():.3e}')