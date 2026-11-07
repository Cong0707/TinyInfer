import json
p = r"E:\Develop\Deepseek\tinyinfer\cache\weights\tokenizer.json"
d = json.load(open(p, encoding="utf-8"))
added = [t.get("content") for t in d.get("added_tokens", [])]
print("added tokens:", len(added))
for t in added[:40]:
    print(" ", repr(t))
