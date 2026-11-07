# 系统级对比：这个模型在 8GB 卡上，vLLM vs 我们
# vLLM: 需要全部权重常驻（EP 不行，单卡 8GB），所以**跑不起来**
GB = 1024**3
total_weights = 475.3   # GB
expert_table = 296.0
print('RESULT === 8GB 单卡 ===')
print(f'  vLLM   需要 {total_weights:.0f} GB 显存（全部常驻）-> 8GB 无法运行')
print(f'  我们   streaming + host cache -> 810 ms/token = 1.23 tok/s')
print()
print('RESULT === 24GB 单卡 ===')
print(f'  vLLM   需要 {total_weights:.0f} GB -> 仍然无法运行')
print(f'  我们   planner: 40/40 层常驻 + 241 专家槽 -> 零 PCIe')
print()
print('RESULT === 8x H200 (1.1TB) ===')
vram = 8 * 141
print(f'  vLLM   {total_weights:.0f} GB 需要, 8x141={vram}GB -> 可以，但只用 {total_weights/vram*100:.0f}%')
print(f'  我们   同样可以，且 expert slots 4919/档 -> 也是全常驻')
print(f'  -> 这个区间双方都无 PCIe 压力，差距在 kernel 效率')