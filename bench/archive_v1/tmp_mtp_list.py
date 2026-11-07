import json
idx = json.load(open(r"E:\Develop\Deepseek\tinyinfer\cache\weights\model.safetensors.index.json", encoding="utf-8"))
wm = idx["weight_map"]
mtp = sorted(k for k in wm if k.startswith("mtp."))
print("mtp tensors:", len(mtp))
for k in mtp:
    print(" ", k, "->", wm[k])
