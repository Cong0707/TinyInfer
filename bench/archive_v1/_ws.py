from pathlib import Path
import json
p = Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\weights.py")
txt = p.read_text(encoding="utf-8")
i = txt.index("class ShardStore")
print(txt[i:i+3000])
