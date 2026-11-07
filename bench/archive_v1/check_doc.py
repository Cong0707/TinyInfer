import re
p = r"E:\Develop\Deepseek\tinyinfer\docs\kernels.md"
c = open(p, encoding="utf-8").read()
print("chars", len(c), "lines", c.count(chr(10))+1)
print("sentinel", c.count(chr(166)), "backslash_backtick", c.count(chr(92)+chr(96)), "fences", c.count(chr(96)*3))
print([l for l in c.splitlines() if l.startswith("## ")])
print([l for l in c.splitlines() if l.startswith("### 2.1")])
