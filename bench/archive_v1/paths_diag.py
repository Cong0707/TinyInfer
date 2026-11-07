import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(19)))
for i in (8, 9):
    b = eng.block(i)
    paths = b._weight_paths()
    print(f"L{i} ({b.attn.__class__.__name__}): {len(paths)} paths")
    print("  ", [(name, tuple(t.shape)) for (o, name, t) in paths][:6], "...")
    print("  ", [name for (o, name, t) in paths][-4:])
b = eng.block(8)
print("block8 device:", eng._layer_device(b), "attn:", b.attn.__class__.__name__)
