import json, struct
p = r"E:\Develop\Deepseek\tinyinfer\cache\weights\model-00044-of-00048.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
for k in ["mtp.0.main_proj.weight", "mtp.0.main_proj.scale", "mtp.0.main_norm.weight",
          "mtp.0.ffn.gate.weight", "mtp.0.attn.wq_a.weight"]:
    info = hdr.get(k)
    if info:
        print(k, info["dtype"], info["shape"])
