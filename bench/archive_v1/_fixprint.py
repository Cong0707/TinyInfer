import io
p = r"E:\Develop\Deepseek\tinyinfer\repl.py"
s = io.open(p, encoding="utf-8").read()
bad = '                    print(f"'
'[server error] {msg}")'
good = '                    print(f"[server error] {msg}")'
assert bad in s, "broken block not found"
s = s.replace(bad, good, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("fixed")