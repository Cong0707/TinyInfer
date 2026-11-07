
import json, pathlib, re

root = pathlib.Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer")
moved = json.loads((root/"_moved.json").read_text(encoding="utf-8"))
# 旧名 -> 新包（"deepseek"/"qwen"/None=仍在根）
where = {k: v for k, v in moved.items()}

files = []
files += list(root.glob("*.py"))
files += list((root/"deepseek").glob("*.py"))
files += list((root/"qwen").glob("*.py"))

pat = re.compile(r"^(\s*)from\s+(\.+)([A-Za-z_][\w.]*)?\s+import\s+(.+)$", re.M)

def pkg_of(path: pathlib.Path) -> str:
    if path.parent.name in ("deepseek", "qwen"):
        return path.parent.name
    return ""

changed = 0
for f in files:
    if f.name in ("__init__.py",):
        continue
    pkg = pkg_of(f)
    txt = f.read_text(encoding="utf-8")
    orig = txt

    def repl(m):
        indent, dots, mod, names = m.group(1), m.group(2), m.group(3) or "", m.group(4)
        # 旧位置一律在包根：'.X' 表示 tinyinfer.X
        if len(dots) != 1:
            return m.group(0)
        head = mod.split(".")[0] if mod else ""
        target = where.get(head)
        if mod and target is None and head not in where:
            # 目标还在根
            if pkg == "":
                return m.group(0)              # 根->根：不变
            return f"{indent}from ..{mod} import {names}"
        if mod:
            if target is None:
                return m.group(0)
            if target == pkg:
                return m.group(0)              # 同包：不变
            if pkg == "":
                return f"{indent}from .{target}.{mod} import {names}"
            return f"{indent}from ..{target}.{mod} import {names}"
        # 'from . import X'（X 是模块名列表）
        parts = [n.strip().split(" as ")[0].strip() for n in names.split(",")]
        outs = []
        for p in parts:
            t = where.get(p)
            if t is None:
                outs.append(p if pkg == "" else p)
            else:
                outs.append(f"{t}.{p}" if pkg == "" else f"{target}.{p}")
        if pkg == "":
            new = [f".{x}" if (x.split('.')[0] in where) else x for x in outs]
        else:
            new = [f"..{x}" if ("." in x and x.split('.')[0] in where) or x.split('.')[0] not in where
                   else x for x in outs]
        flat = ", ".join(x if x.startswith(".") else f".{x}" for x in new)
        return f"{indent}from . import {names}" if flat == f". " else f"{indent}from {'..' if pkg else '.'} import {local}" if False else m.group(0)

    txt = pat.sub(repl, txt)
    # 绝对 import tinyinfer.X
    for old, tgt in where.items():
        txt = txt.replace(f"from tinyinfer.{old} import", f"from tinyinfer.{tgt}.{old} import")
        txt = txt.replace(f"import tinyinfer.{old}\b", f"import tinyinfer.{tgt}.{old}")
    if txt != orig:
        f.write_text(txt, encoding="utf-8")
        changed += 1
print("rewrote:", changed, "/", len(files))
