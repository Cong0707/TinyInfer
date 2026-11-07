p = r"E:\Develop\Deepseek\tinyinfer\docs\kernels.md"
c = open(p, encoding="utf-8").read()
print("doc chars", len(c), "lines", c.count(chr(10))+1, "bs_bt", c.count(chr(92)+chr(96)))
print([l for l in c.splitlines() if l.startswith("## ")])
