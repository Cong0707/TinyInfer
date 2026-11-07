import json
idx = json.load(open(r"E:\Develop\Deepseek\tinyinfer\cache\weights\model.safetensors.index.json", encoding="utf-8"))
wm = idx["weight_map"]
n = [k for k, v in wm.items() if "00001" in v]
print("tensors in 00001:", len(n))
for k in n[:12]:
    print(" ", k)
