import re
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\loading.py"
src = open(p, encoding="utf-8").read().splitlines()
for i, l in enumerate(src, 1):
    if "_materialize(" in l and "def " not in l:
        print(i, "|", l.strip())