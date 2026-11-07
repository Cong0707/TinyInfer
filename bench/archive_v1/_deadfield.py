
import ast, os, re, sys
from pathlib import Path

root = Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer")
tiers = root / "tiers.py"

# 1) 取 TierConfig 的全部字段
src = tiers.read_text(encoding="utf-8")
tree = ast.parse(src)
fields = []
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "TierConfig":
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fields.append(stmt.target.id)
print(f"TierConfig 字段数: {len(fields)}")

# 2) 扫所有 .py，找字段被读取的地方（属性访问 .name，排除 tiers.py 的定义行和构造参数）
pyfiles = [p for p in root.rglob("*.py") if "__pycache__" not in str(p)]
readers = {f: [] for f in fields}
for p in pyfiles:
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    try:
        t = ast.parse(text)
    except SyntaxError:
        continue
    for node in ast.walk(t):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.attr in readers:
                # 排除 tiers.py 里构造时的关键字（那是 Call 的 keyword，不是 Attribute）
                readers[node.attr].append((p.name, node.lineno))
        # 也统计 getattr(cfg, "x") 和 cfg["x"]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                if node.args[1].value in readers:
                    readers[node.args[1].value].append((p.name, node.lineno))
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value in readers:
                readers[node.slice.value].append((p.name, node.lineno))

print(f"\n{'field':<26} {'readers':>7}  where")
print("-"*78)
dead = []
for f in fields:
    rs = [(a,b) for a,b in readers[f] if not (a == "tiers.py")]
    # tiers.py 内部的读取也算，但要区分「定义处」
    allr = readers[f]
    internal = [x for x in allr if x[0] == "tiers.py"]
    external = [x for x in allr if x[0] != "tiers.py"]
    n = len(external)
    where = ", ".join(sorted({a for a,_ in external}))[:44] if external else ""
    flag = "" if n else "  <-- 外部无读者"
    if not n: dead.append((f, len(internal)))
    print(f"{f:<26} {n:>7}  {where}{flag}")

print(f"\n=== 只写不读（tiers.py 之外没人读）: {len(dead)} 个 ===")
for f, ni in dead:
    print(f"  {f}   (tiers.py 内部读取 {ni} 次)")
