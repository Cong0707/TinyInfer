import json
cfg = json.load(open(r"E:\Develop\Deepseek\tinyinfer\cache\weights\config.json", encoding="utf-8"))
def walk(d, prefix=""):
    for k, v in d.items():
        if k.startswith("dspark") or k in ("n_activated_experts", "n_routed_experts", "hc_mult", "dim", "n_layers"):
            print(prefix + k, "=", v if not isinstance(v, (dict, list)) or len(str(v)) < 80 else str(v)[:80])
walk(cfg.get("text_config", cfg))
print("--- top-level keys:", list(cfg.keys())[:20])
