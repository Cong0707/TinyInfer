
import sys, dataclasses
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.tiers import plan, OptLevel
a, b = plan(OptLevel.BALANCED), plan(OptLevel.AGGRESSIVE)
print(f"{'field':<26} {'BALANCED':<24} {'AGGRESSIVE':<24}")
print("-"*92)
for f in dataclasses.fields(a):
    va, vb = getattr(a, f.name), getattr(b, f.name)
    mark = "" if va == vb else "  <<<"
    print(f"{f.name:<26} {str(va)[:22]:<24} {str(vb)[:22]:<24}{mark}")
