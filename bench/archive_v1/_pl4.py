import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.resources import model_profile, plan, Hardware, Card
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
prof = model_profile(d)
print('RESULT layer=%.0fMB expert=%.1fMB mtp=%.0fMB' % (prof['layer_bytes']/2**20, prof['expert_bytes']/2**20, prof.get('mtp_bytes',0)/2**20))
GB = 1024**3
for name, vram in (('RTX5060 8G', 8.0), ('RTX4090 24G', 24.0), ('RTX5090 32G', 32.0), ('H100 80G', 80.0), ('H200 141G', 141.0)):
    hw = Hardware(name=name, vram_gb=vram, ram_gb=47.0, ram_avail_gb=30.0,
                  cards=[Card(name=name, total_gb=vram, free_gb=vram*0.95, bandwidth_gbps=2000)])
    try:
        p = plan(hw=hw, n_layers=40, model=prof, speculation=True)
        print(f'RESULT {name:12s} layers={p.gpu_layers}/40 stream={p.stream_layers} slots={p.slots} cache={p.cache_bytes/GB:.1f}G spec={p.speculation}')
    except Exception as e:
        print(f'RESULT {name}: {type(e).__name__} {e}'[:130])