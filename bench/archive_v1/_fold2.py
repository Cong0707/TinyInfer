from pathlib import Path
import re
base = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc")

def fold_ifdef(lines):
    out, i, n = [], 0, 0
    while i < len(lines):
        s = lines[i].strip()
        if re.fullmatch(r"#\s*ifdef\s+USE_ROCM", s):
            depth, j, body, alt, cur = 1, i + 1, [], [], None
            cur = body
            has_else = False
            while j < len(lines) and depth > 0:
                st = lines[j].strip()
                if re.match(r"#\s*if", st):
                    depth += 1
                elif re.match(r"#\s*endif", st):
                    depth -= 1
                    if depth == 0:
                        break
                elif re.match(r"#\s*else", st) and depth == 1:
                    cur = alt; has_else = True; j += 1; continue
                cur.append(lines[j]); j += 1
            out.extend(alt if has_else else [])
            n += 1; i = j + 1
        else:
            out.append(lines[i]); i += 1
    return out, n

tf, tn = 0, 0
for f in list(base.rglob("*.cu")) + list(base.rglob("*.cuh")):
    txt = f.read_text(encoding="utf-8", errors="replace")
    if "USE_ROCM" not in txt:
        continue
    lines = txt.splitlines(keepends=True)
    n_all = 0
    for _ in range(8):
        lines2, n = fold_ifdef(lines)
        if n == 0:
            break
        lines = lines2; n_all += n
    if n_all:
        f.write_text("".join(lines), encoding="utf-8")
        print("ifdef folded", n_all, f.relative_to(base))
        tf += 1; tn += n_all
print("files:", tf, "folds:", tn)