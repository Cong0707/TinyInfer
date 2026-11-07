import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.loading import VramSlotCache

blob = 1 << 20
print('RESULT === 生命周期（正确用法：ready=False 时 copy_in+publish）===')
c = VramSlotCache(4, blob, device='cuda:0')
slot, ready = c.acquire(0, 42)
print(f'  acquire(0,42) -> slot={slot} ready={ready}  (False = 需要 copy_in)')
data = torch.full((blob,), 0xAB, dtype=torch.uint8)
c.copy_in(slot, data)
c.publish(slot)
buf = c.buffer(slot)
print(f'  after publish: buf[0]=0x{buf[0].item():02X} all_same={(buf==0xAB).all().item()}')
c.release(0, 42)
slot2, ready2 = c.acquire(0, 42)
print(f'  re-acquire -> slot={slot2} ready={ready2}  (True = 命中，数据已就绪)')
buf2 = c.buffer(slot2)
print(f'  data intact: {(buf2==0xAB).all().item()}')
c.release(0, 42)
print(f'  stats: {c.stats()}')

print('RESULT === 淘汰 ===')
c2 = VramSlotCache(2, blob, device='cuda:0')
for e in (1, 2):
    s, n = c2.acquire(0, e)
    c2.copy_in(s, torch.full((blob,), e, dtype=torch.uint8))
    c2.publish(s)
    c2.release(0, e)
print(f'  filled 2 slots, staged={c2.staged_count}')
s3, n3 = c2.acquire(0, 3)
print(f'  acquire(0,3) -> slot={s3} ready={n3} evictions={c2.evictions}')
print(f'  has(0,1)={c2.has(0,1)} has(0,2)={c2.has(0,2)} has(0,3)={c2.has(0,3)}')
print(f'  stats: {c2.stats()}')