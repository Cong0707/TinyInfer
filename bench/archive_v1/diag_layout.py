import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

eng = Engine(r"E:\Develop\Deepseek\tinyinfer\cache\weights", tier=OptLevel.BALANCED,
             device="cuda", layers=list(range(26)))

def sig(i):
    b = eng.block(i)
    return [(n, tuple(t.shape), str(t.dtype), t.numel() * t.element_size())
            for (_, n, t) in b._weight_paths()]

# 双缓冲的 kind 判定与 layout 来源
for first, cur in ((7, 8), (9, 10), (21, 22), (24, 25)):
    s1, s2 = sig(first), sig(cur)
    diffs = []
    for k, (a, b) in enumerate(zip(s1, s2)):
        if a[1] != b[1] or a[2] != b[2] or a[3] != b[3]:
            diffs.append((k, a, b))
    print(f"first=L{first} cur=L{cur}: n={len(s1)}/{len(s2)} diffs={len(diffs)}")
    for d in diffs[:5]:
        print("   ", d)
