
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
for tier in (OptLevel.BALANCED, OptLevel.AGGRESSIVE, OptLevel.EXTREME):
    try:
        eng = Engine(d, tier=tier, device="cuda", layers=list(range(4)))
        print(f"[{tier.name}] OK  kv_quant={eng._kv_quant} kv_eff={eng._kv_bits_effective} "
              f"skip={sorted(eng._skip_layers)[:3]}... prefetch={eng.enable_prefetch} "
              f"fast_math={bool(eng._fast_math)}")
        del eng
        import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"[{tier.name}] FAILED: {type(e).__name__}: {str(e)[:200]}")
