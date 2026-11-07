# 纯算术：每个 token 必须搬多少
LAYERS = 40
LAYER_B = 204 * 1024**2      # 一层全部权重（含 attention + shared expert，不含 routed）
EXPERT_B = 19.7 * 1024**2    # 一个 routed expert
TOPK = 6
EXPERTS_PER_TOKEN = LAYERS * TOPK   # 240
ROUTED_B = EXPERTS_PER_TOKEN * EXPERT_B
print(f'routed experts: {EXPERTS_PER_TOKEN} x {EXPERT_B/1e6:.1f}MB = {ROUTED_B/2**30:.2f} GB')
print(f'all 40 layers  : 40 x {LAYER_B/1e6:.0f}MB = {LAYERS*LAYER_B/2**30:.2f} GB')
print(f'全部：{(ROUTED_B + LAYERS*LAYER_B)/2**30:.2f} GB 每 token')
print()
# 但专家去重后：384 个专家，一层最多 384，一个 token 打 6 个
UNIQ = 384
print(f'一层所有专家: {UNIQ} x {EXPERT_B/1e6:.1f}MB = {UNIQ*EXPERT_B/2**30:.2f} GB')
print(f'40 层所有专家: {LAYERS*UNIQ*EXPERT_B/2**30:.2f} GB  <-- 模型专家总大小')
print()
for name, cap_gb, bw_gbs in (('RTX5060 8GB', 6.9, 26.5), ('RTX5090 32GB', 30, 1792), ('H100 80GB', 76, 3350), ('H200 141GB', 137, 4800)):
    cap = cap_gb * 2**30
    layer_part = LAYERS * LAYER_B
    if cap >= layer_part + ROUTED_B:
        print(f'{name:14s} 全部常驻 -> compute-bound')
    else:
        miss = (layer_part + ROUTED_B) - cap
        print(f'{name:14s} 缺 {miss/2**30:.1f}GB -> 传输下界 {miss/bw_gbs/1e9*1000:.1f} ms ({(1/(miss/bw_gbs/1e9))*1000/1000:.0f} tok/s 上限)')