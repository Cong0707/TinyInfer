import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.resources import plan, Hardware, Card, model_profile
from tinyinfer.loading import ShardIndex
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
prof = model_profile(ShardIndex(d))
GB = 1<<30
print('RESULT 找最小能零 PCIe 的显存')
for vram in (16, 18, 20, 22, 23, 24):
    hw = Hardware(name=f'{vram}G', vram_gb=float(vram), ram_gb=64.0, ram_avail_gb=50.0,
                  cards=[Card(0, f'{vram}G', float(vram), vram*0.95*GB)])
    p = plan(hw=hw, n_layers=40, model=prof, speculation=False)  # 先不投机
    ok = p.expert_slots >= 240
    print(f'RESULT {vram:2d}GB: layers={p.gpu_layers:2d}/40 slots={p.expert_slots:4d} {"ZERO-PCIE" if ok else "short by "+str(240-p.expert_slots)}')