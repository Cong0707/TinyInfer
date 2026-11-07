import json
idx = json.load(open(r"E:\Develop\Deepseek\tinyinfer\cache\weights\model.safetensors.index.json", encoding="utf-8"))
wm = idx["weight_map"]
mtp = sorted(k for k in wm if k.startswith("mtp."))
special = [k for k in mtp if ".ffn.experts." not in k]
print("non-expert mtp tensors:", len(special))
for k in special:
    print(" ", k)
print()
# per-stage counts
import collections
c = collections.Counter(k.split(".")[1] for k in mtp)
print("per stage:", dict(c))
