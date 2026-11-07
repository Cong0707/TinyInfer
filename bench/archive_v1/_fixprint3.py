import io
p = r"E:\Develop\Deepseek\tinyinfer\repl.py"
rows = io.open(p, encoding="utf-8").read().split(chr(10))
# lines 114-115 are one broken f-string spread over two rows (1-based)
assert rows[113].rstrip().endswith('print(f"'), rows[113]
assert rows[114].strip().startswith("[server error]"), rows[114]
rows[113:115] = ['                    print("[server error] " + str(msg))']
io.open(p, "w", encoding="utf-8", newline="").write(chr(10).join(rows))
print("fixed")