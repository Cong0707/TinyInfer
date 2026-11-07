# 我们的引擎在什么硬件上能跑，vLLM 不能？
LAYERS = 40
LAYER_B = 204 * 1024**2
EXPERT_B = 19.7 * 1024**2
TOPK = 6
ROUTED = LAYERS * TOPK * EXPERT_B
TOTAL_W = LAYERS * LAYER_B + 40*384*EXPERT_B
print(f'模型全部权重: {TOTAL_W/2**30:.1f} GB')
print(f'vLLM 需要:     >= {TOTAL_W/2**30:.0f} GB 显存（全部专家必须在卡上）')
print()
print('我们的引擎（流式 + 分片）能跑的配置：')
for name, vram in (('RTX 5060 8GB', 6.9), ('RTX 4090 24GB', 22), ('RTX 5090 32GB', 30), ('H100 80GB', 76), ('H200 141GB', 137)):
    cap = vram * 2**30
    # 每 token 必须传输的量
    stream_bytes = max(0, LAYERS*LAYER_B + ROUTED - cap)
    print(f'  {name:14s} 每token传输 {stream_bytes/2**30:5.1f}GB')
print()
print('vLLM: 上面全部跑不了（需要 300GB+ 显存）')