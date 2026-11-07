import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py"
t = io.open(p, encoding="utf-8").read()
old = """                dec.replay_step()
                tok_id = sampler.pick(dec.logits_buf, produced)"""
new = """                dec.replay_step()
                tok_id = sampler.pick(dec.logits_buf, produced)
                dec.set_token(tok_id)"""
assert old in t and "dec.set_token(tok_id)" not in t
t = t.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("patched")
