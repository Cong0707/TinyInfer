import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.resources import model_profile, plan, Hardware, Card
from tinyinfer.loading import ShardIndex
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
prof = model_profile(ShardIndex(d))
GB = 1024**3
hw = Hardware(name='8G', vram_gb=8.0, ram_gb=47.0, ram_avail_gb=30.0,
              cards=[Card(0, '8G', 8.0, 7.6)])
p = plan(hw=hw, n_layers=40, model=prof, speculation=False)
print(f'RESULT layers={p.gpu_layers} stream={p.stream_layers} slots={p.expert_slots} cache={p.cache_bytes/GB:.1f}G')
print(f'RESULT reason={p.reason[:200] if p.reason else None}')