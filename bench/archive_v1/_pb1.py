import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.resources import model_profile, plan, Hardware, Card
from tinyinfer.loading import ShardIndex
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
prof = model_profile(ShardIndex(d))
GB = 1024**3
cases = [
    ('tiny 2GB', 2.0, 8.0, 6.0),
    ('4GB', 4.0, 16.0, 12.0),
    ('8GB', 8.0, 47.0, 30.0),
    ('16GB', 16.0, 64.0, 50.0),
    ('24GB', 24.0, 64.0, 50.0),
    ('1TB', 1024.0, 2048.0, 1800.0),
    ('4x24GB', 24.0, 128.0, 100.0),
]
for name, vram, ram, avail in cases:
    hw = Hardware(name=name, vram_gb=vram, ram_gb=ram, ram_avail_gb=avail,
                  cards=[Card(0, name, vram, vram*0.95)])
    try:
        p = plan(hw=hw, n_layers=40, model=prof, speculation=True)
        ok = (0 <= p.gpu_layers <= 40 and p.stream_layers >= 0 and p.expert_slots >= 0
              and p.cache_bytes >= 0)
        print(f'RESULT {name:10s} layers={p.gpu_layers:2d}/40 stream={p.stream_layers:2d} '
              f'slots={p.expert_slots:5d} cache={p.cache_bytes/GB:6.1f}G fits={p.fits} ok={ok}')
        if not ok:
            print(f'  !! INVARIANT BROKEN: {p.reason[:120]}')
    except Exception as e:
        print(f'RESULT {name:10s} RAISED {type(e).__name__}: {e}'[:130])