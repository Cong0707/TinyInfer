import json, struct
p = r"E:\Models\DeepSeek-V4.1-Flash-1017\model-00047-of-00048.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    print("header size:", n)
    total = max(e["data_offsets"][1] for e in hdr.values())
    print("max data offset:", total, "GB:", round(total / 2**30, 2))
    for k, e in sorted(hdr.items()):
        print(" ", k, e["dtype"], e["shape"], "bytes:", e["data_offsets"][1] - e["data_offsets"][0])
