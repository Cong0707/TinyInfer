import json
idx = json.load(open(r"E:\Develop\Deepseek\tinyinfer\cache\weights\model.safetensors.index.json", encoding="utf-8"))
wm = idx["weight_map"]
names = [k for k in wm if "engram" in k]
print("engram tensors:", len(names))
for k in names: print(" ", k, "->", wm[k])
