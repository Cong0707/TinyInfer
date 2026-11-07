
import re, pathlib

root = pathlib.Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer")
m = root / "model.py"
s = m.read_text(encoding="utf-8")
# keep the helper's own definition intact; rewrite every call site
n_before = s.count("setattr(owner, name")
s2 = s.replace("setattr(owner, name,", "set_weight(owner, name,")
s2 = s2.replace("setattr(owner, name,\n", "set_weight(owner, name,\n")
m.write_text(s2, encoding="utf-8")
print("model.py call sites:", n_before, "->", s2.count("setattr(owner, name"), "left")

e = root / "engine.py"
t = e.read_text(encoding="utf-8")
n_before2 = t.count("setattr(owner, name")
t2 = t.replace("setattr(owner, name,", "set_weight(owner, name,")
t2 = t2.replace("setattr(owner, name,\n", "set_weight(owner, name,\n")
e.write_text(t2, encoding="utf-8")
print("engine.py call sites:", n_before2, "->", t2.count("setattr(owner, name"), "left")
print("engine imports model?", "from .model import" in t2)
for line in t2.splitlines():
    if "from .model import" in line:
        print("  ", line.strip())
