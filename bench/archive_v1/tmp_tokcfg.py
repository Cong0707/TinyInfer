import json
p = r"E:\Develop\Deepseek\tinyinfer\cache\weights\tokenizer_config.json"
d = json.load(open(p, encoding="utf-8"))
print("keys:", list(d.keys())[:20])
print("chat_template:", str(d.get("chat_template"))[:600])
