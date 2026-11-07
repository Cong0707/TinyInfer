# 8GB 实测: 790ms/token, fetch 285ms(36%), budget 80ms
# 20GB: 零 PCIe, 所以 fetch 应该在理论上降到 ~74ms (截距，即固定开销)
print('RESULT === 20GB 的理论推算 ===')
print('  8GB:  fetch 285ms  = 114us/MB x 1852MB + 74ms')
print('  20GB: 1848 个不同专家全部常驻(247槽 > 240 需求)')
print('        => 稳态下每 token 读取 0 MB')
print('        => fetch = 74ms (截距，即固定开销)')
print()
base = 790.0
fetch_8 = 285.0
fetch_20 = 74.0
print(f'  total: {base:.0f} -> {base - (fetch_8-fetch_20):.0f} ms = {1000/(base-(fetch_8-fetch_20)):.2f} tok/s')
print(f'  提升: {1000/(base-(fetch_8-fetch_20))/(1000/base)-1:.0%}')
print()
print('RESULT === 但这是乐观估计 ===')
print('  247 槽刚好覆盖 240 需求，没有余量')
print('  缓存命中率依赖路由重复，20GB 下槽池是 VRAM 而非 RAM cache')
print('  实际可能有 eviction，需要实测才能确认')