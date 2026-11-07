import json
cfg = json.load(open(r"E:\Develop\Deepseek\tinyinfer\cache\weights\config.json", encoding="utf-8"))
tc = cfg.get("text_config", cfg)
for k in ("dspark_target_layer_ids", "dspark_block_size", "dspark_n_routed_experts",
          "dspark_n_activated_experts", "n_mtp_layers", "hc_mult", "dim", "n_layers"):
    print(k, "=", tc.get(k))
