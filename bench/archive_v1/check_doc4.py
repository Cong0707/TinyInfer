p = r"E:\Develop\Deepseek\tinyinfer\docs\kernels.md"
c=open(p,encoding="utf-8").read()
print("chars",len(c),"bs_bt",c.count(chr(92)+chr(96)))
print([l for l in c.splitlines() if l.startswith("### 2.14") or l.startswith("## ")])
