from pathlib import Path
import re
base = Path(r"E:/Develop/Deepseek/repositories/vllm-1c879919786497217b3b5fb0258d1d72f2442cb0/csrc")

def fold_once(lines):
    out, i, n = [], 0, 0
    while i < len(lines):
        s = lines[i].strip()
        is_ifndef = bool(re.fullmatch(r"#\s*ifndef\s+USE_ROCM", s))
        is_ifnotdef = bool(re.fullmatch(r"#\s*if\s+!\s*defined\s*\(?\s*USE_ROCM\s*\)?", s))
        if is_ifndef or is_ifnotdef:
            depth, j, body, alt, cur = 1, i + 1, [], [], None
            cur = body
            while j < len(lines) and depth > 0:
                st = lines[j].strip()
                if re.match(r"#\s*if", st):
                    depth += 1
                elif re.match(r"#\s*endif", st):
                    depth -= 1
                    if depth == 0:
                        break
                elif re.match(r"#\s*else", st) and depth == 1:
                    cur = alt; j += 1; continue
                cur.append(lines[j]); j += 1
            out.extend(body if body else alt)
            n += 1; i = j + 1
        else:
            out.append(lines[i]); i += 1
    return out, n

total_files, total_folds = 0, 0
for f in list(base.rglob("*.cu")) + list(base.rglob("*.cuh")):
    txt = f.read_text(encoding="utf-8", errors="replace")
    if "USE_ROCM" not in txt:
        continue
    lines = txt.splitlines(keepends=True)
    n_all = 0
    for _ in range(8):
        lines2, n = fold_once(lines)
        if n == 0:
            break
        lines = lines2; n_all += n
    if n_all:
        f.write_text("".join(lines), encoding="utf-8")
        print("folded", n_all, f.relative_to(base))
        total_files += 1; total_folds += n_all
print("files:", total_files, "folds:", total_folds)