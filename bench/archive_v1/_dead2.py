
import ast
from pathlib import Path
root = Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer")
# 收集所有 dataclass 字段
targets = {}
for p in root.rglob("*.py"):
    if "__pycache__" in str(p): continue
    try: t = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError: continue
    for node in ast.walk(t):
        if isinstance(node, ast.ClassDef):
            fs = []
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    fs.append(st.target.id)
            if fs: targets.setdefault(node.name, set()).update(fs)

# 全仓文本扫描
alltext = {}
for p in root.rglob("*.py"):
    if "__pycache__" in str(p): continue
    try: alltext[p] = p.read_text(encoding="utf-8", errors="ignore")
    except Exception: pass
blob = "\n".join(alltext.values())

print("=== dataclass 里疑似只写不读的字段 ===")
for cls, fs in sorted(targets.items()):
    dead = []
    for f in fs:
        # 该字段名在类定义文件之外出现次数
        cnt = 0
        for p, txt in alltext.items():
            for node in ast.walk(ast.parse(txt)) if False else []:
                pass
        # 简单文本计数（排除 def/ann 行）
        import re
        hits = [m for m in re.finditer(rf"\b{re.escape(f)}\b", blob)]
        # 若只在 tiers-like 定义处出现（构造参数 + 注解），算死
        if len(hits) <= 2:
            dead.append((f, len(hits)))
    if dead:
        print(f"{cls}: {', '.join(f'{f}({n})' for f,n in dead)}")
