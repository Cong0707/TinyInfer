import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.loading import VramSlotCache

print('RESULT === VramSlotCache 状态机 ===')
blob = 1 << 20  # 1MB 测试用
for n_slots in (1, 4, 16):
    c = VramSlotCache(n_slots, blob, device='cuda:0')
    print(f'  slots={n_slots} state_len={len(c._state)} all_free={all(s==c.FREE for s in c._state)}')
    print(f'    total_slots={c.total_slots} total_bytes={c.total_bytes}')

print('RESULT === 分配/释放循环 ===')
c = VramSlotCache(4, blob, device='cuda:0')
# 检查公开 API
api = [a for a in dir(c) if not a.startswith('_')]
print('  API:', api)
