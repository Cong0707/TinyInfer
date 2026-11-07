import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.resources import model_profile, plan, Hardware, Card
from tinyinfer.weights import ShardIndex
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
idx = ShardIndex(d)
prof = model_profile(idx)
print('RESULT topk:', prof.get('topk'))
GB = 1024**3
print(f'{"card":14s} {"layers":>8s} {"stream":>7s} {"slots":>7s} {"need":>6s} {"cache":>7s} {"spec":>5s}')
need = int(prof.get('topk') or 0) * 40
for name, vram in (('RTX5060 8G', 8.0), ('RTX4090 24G', 24.0), ('RTX5090 32G', 32.0), ('H100 80G', 80.0), ('H200 141G', 141.0)):
    hw = Hardware(name=name, vram_gb=vram, ram_gb=64.0, ram_avail_gb=56.0,
                  cards=[Card(0, name, vram, vram*0.95)])
    p = plan(hw=hw, n_layers=40, model=prof, speculation=True)
    print(f'RESULT {name:14s} {p.gpu_layers:3d}/40 {p.stream_layers:7d} {p.expert_slots:7d} {need:6d} {p.cache_bytes/GB:6.1f}G {str(p.speculation):>5s}')