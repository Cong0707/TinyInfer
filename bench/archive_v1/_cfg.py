import json
p = r"E:\Models\Qwen3.5-0.8B\config.json"
d = json.load(open(p, encoding="utf-8"))
print("=== sampling fields in config.json ===")
for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
          "do_sample", "eos_token_id", "bos_token_id", "pad_token_id"):
    print(f"  {k}: {d.get(k, '<absent>')}")
tc = d.get("text_config") or {}
print("=== in text_config ===")
for k in ("temperature", "top_p", "top_k", "eos_token_id"):
    if k in tc:
        print(f"  {k}: {tc[k]}")
print("=== full top-level keys ===")
print(sorted(d.keys()))
