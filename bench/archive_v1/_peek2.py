import io
p = r"E:\Develop\Deepseek\tinyinfer\repl.py"
rows = io.open(p, encoding="utf-8").read().split(chr(10))
for i in range(106, 122):
    print(i+1, repr(rows[i]))