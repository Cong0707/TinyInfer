import json
idx = json.load(open(r"E:\Develop\Deepseek\tinyinfer\cache\weights\model.safetensors.index.json", encoding="utf-8"))
wm = idx["weight_map"]
hc = [k for k in wm if "hc_head" in k or ".hc_" in k]
print("hc tensors:", len(hc))
for k in hc:
    print(" ", k)
