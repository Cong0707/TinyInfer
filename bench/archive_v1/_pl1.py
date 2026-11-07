import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.resources import plan, Hardware, Card, model_profile
from tinyinfer.loading import ShardIndex
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
prof = model_profile(ShardIndex(d))
GB = 1<<30
print('RESULT card      layers stream slots  mtp  cacheGB  expertsNeeded')
for vram, name in ((24,'RTX4090'), (32,'RTX5090'), (80,'H100'), (141,'H200')):
    hw = Hardware(name=name, vram_gb=float(vram), ram_gb=256.0, ram_avail_gb=230.0,
                  cards=[Card(0, name, float(vram), vram*0.95*GB)])
    p = plan(hw=hw, n_layers=40, model=prof, speculation=True)
    need = len(p.gpu_layers) if hasattr(p,'gpu_layers') and isinstance(p.gpu_layers,list) else p.gpu_layers
    # 240 个不同专家/token 需要多少个槽
    need_slots = 240
    ok = p.expert_slots >= need_slots
    print(f'RESULT {name:9s} {p.gpu_layers:2d}/40  {p.stream_layers:2d}  {p.expert_slots:5d}  {str(p.speculation):5s} {p.cache_bytes/GB:6.1f}  {"OK" if ok else "SHORT"}')