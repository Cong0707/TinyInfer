import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.loading import VramSlotCache

blob = 1 << 20
c = VramSlotCache(4, blob, device='cuda:0')
print('RESULT === 生命周期 ===')
slot, is_new = c.acquire(0, 42)
print(f'  acquire(0,42) -> slot={slot} is_new={is_new}')
data = torch.arange(blob // 4, dtype=torch.int32)
c.copy_in(slot, data.view(torch.uint8))
c.publish(slot)
buf = c.buffer(slot)
print(f'  copy_in+publish: buf[:4]={buf[:4].tolist()}')
c.release(0, 42)
print(f'  released, has(0,42)={c.has(0,42)}')

print('RESULT === 命中 ===')
slot2, is_new2 = c.acquire(0, 42)
print(f'  re-acquire(0,42) -> slot={slot2} is_new={is_new2} (should be same slot, not new)')
c.release(0, 42)
print(f'  stats: {c.stats()}')

print('RESULT === 淘汰 ===')
c2 = VramSlotCache(2, blob, device='cuda:0')
for e in (1, 2):
    s, n = c2.acquire(0, e)
    c2.copy_in(s, torch.full((blob,), e, dtype=torch.uint8))
    c2.publish(s)
    c2.release(0, e)
print(f'  filled 2 slots, staged={c2.staged_count()}')
s3, n3 = c2.acquire(0, 3)
print(f'  acquire(0,3) -> slot={s3} is_new={n3} evictions={c2.evictions}')
print(f'  has(0,1)={c2.has(0,1)} has(0,2)={c2.has(0,2)} has(0,3)={c2.has(0,3)}')