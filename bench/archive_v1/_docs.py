import io, re, pathlib
root = pathlib.Path(r"E:\Develop\Deepseek\tinyinfer\docs")
# the deleted one-off scripts are referenced in the kernel notes; point at what replaced them
repl = {
    "scripts/warm_prefill.py": "(已移出仓库；scripts/bench.py 的 warm_prefill 阶段覆盖同一测量)",
    "scripts/precompile.py": "(已移出仓库；首次 prefill 需要编译，见 scripts/build_native.ps1)",
    "scripts/precompile_long.py": "(已移出仓库；长上下文首次 prefill 前需要预热)",
    "scripts/e2e_models.py": "scripts/check.ps1",
    "e2e_mtp.py": "(已移出仓库)",
    "e2e_spec.py": "(已移出仓库)",
}
n = 0
for f in list(root.glob("*.md")) + list(root.glob("history/*.md")):
    t = io.open(f, encoding="utf-8").read()
    o = t
    for a, b in repl.items():
        t = t.replace(a, b)
    if t != o:
        io.open(f, "w", encoding="utf-8", newline="").write(t)
        n += 1
        print("fixed", f.name)
print("files touched:", n)