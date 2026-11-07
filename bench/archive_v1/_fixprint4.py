import io
p = r"E:\Develop\Deepseek\tinyinfer\repl.py"
rows = io.open(p, encoding="utf-8").read().split(chr(10))
assert rows[114] == '[server error] {msg}")', repr(rows[114])
del rows[114]
io.open(p, "w", encoding="utf-8", newline="").write(chr(10).join(rows))
print("removed stray line")