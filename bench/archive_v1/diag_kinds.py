import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(26)))
for i in (9, 10, 12, 21, 22, 24):
    b = eng.block(i)
    kind = (eng.cfg.compress_ratios[i] > 0, getattr(b.attn, "compressor", None) is not None,
            getattr(b.attn, "indexer", None) is not None)
    paths = b._weight_paths()
    sigs = [(name, tuple(t.shape), str(t.dtype)) for (_, name, t) in paths]
    print(f"L{i} kind={kind} n={len(sigs)}")
    print("  ", sigs[:5], "...", sigs[-3:])
