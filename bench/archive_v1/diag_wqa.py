import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(26)))
for i in (9, 10, 21):
    b = eng.block(i)
    print(f"L{i}:")
    for (owner, name, t) in b._weight_paths():
        if name in ("weight", "scale"):
            print(f"   {name} {tuple(t.shape)} {t.dtype}")
        else:
            print(f"   {name} {tuple(t.shape)} {t.dtype}")
