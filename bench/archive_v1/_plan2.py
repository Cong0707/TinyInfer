import sys
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.tp import Device, Placement, plan_for
# H200 141GB 单卡
for name, vram_gb, bw in (('H200', 141.0, 4800.0), ('H100', 80.0, 3350.0), ('RTX5090', 32.0, 1792.0), ('A100', 80.0, 2039.0)):
    d = Device(name='cuda:0', vram=int(vram_gb*1024**3), bandwidth_gbps=bw)
    try:
        p = plan_for([d], total_layers=40, expert_bytes=18.8*1024**2, layer_bytes=166*1024**2)
        print(f'{name:8s} vram={vram_gb:5.0f}GB layers={p.gpu_layers}/{40} slots={p.slots}')
    except Exception as e:
        print(f'{name:8s} {e!r}')