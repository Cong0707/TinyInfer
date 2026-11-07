
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
# 手动减层，找出能建起来的最大层数
for n in (10, 20, 30):
    try:
        eng = Engine(d, tier=OptLevel.BALANCED, device="cuda", layers=list(range(n)))
        print(f"layers={n}: OK")
        del eng
        import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"layers={n}: FAIL {type(e).__name__}: {str(e)[:90]}")
        break
