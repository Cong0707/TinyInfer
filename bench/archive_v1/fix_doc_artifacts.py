p = r"E:\Develop\Deepseek\tinyinfer\docs\kernels.md"
c = open(p, encoding="utf-8").read()
before = c.count(chr(92)+chr(96))
c = c.replace(chr(92)+chr(96), chr(96))
open(p, "w", encoding="utf-8").write(c)
c2 = open(p, encoding="utf-8").read()
print("fixed", before, "remaining", c2.count(chr(92)+chr(96)), "chars", len(c2))
print([l for l in c2.splitlines() if l.startswith("## 8")])
