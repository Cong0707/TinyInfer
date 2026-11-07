
import json, pathlib, re
root = pathlib.Path(r"E:\Develop\Deepseek\tinyinfer")
pk = root/"src"/"tinyinfer"
moved = json.loads((pk/"_moved.json").read_text(encoding="utf-8"))

targets = list((root/"tests").rglob("*.py")) + list((root/"scripts").rglob("*.py")) \
        + list((root/"tools").rglob("*.py")) + [root/"repl.py"]
n = 0
for f in targets:
    if not f.exists():
        continue
    t = f.read_text(encoding="utf-8"); o = t
    for old, tgt in moved.items():
        # 绝对路径
        t = re.sub(rf"\btinyinfer\.{old}\b", f"tinyinfer.{tgt}.{old}", t)
        # 相对路径 from .X / from tinyinfer.X
    if t != o:
        f.write_text(t, encoding="utf-8"); n += 1
print("rewrote tests/scripts/tools:", n)
