import io
src = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py"
txt = io.open(src, encoding="utf-8").read()

old = '''        dec = GraphDecoder.from_prompt(self._model, ids).capture()
        if params.temperature and params.temperature > 0:
            raise BackendError("qwen backend is greedy-only today (temperature=0)")
        produced = 0'''
new = '''        dec = GraphDecoder.from_prompt(self._model, ids).capture()
        sampler = _Sampler(params, torch)
        produced = 0'''
assert old in txt
txt = txt.replace(old, new, 1)

# every token now comes from the sampler, not from argmax inside the decoder
old2 = """            while produced < budget:
                tok_id = dec.next_token()"""
new2 = """            while produced < budget:
                dec.replay_step()
                tok_id = sampler.pick(dec.logits_buf, produced)"""
assert old2 in txt
txt = txt.replace(old2, new2, 1)

io.open(src, "w", encoding="utf-8", newline="").write(txt)
print("patched generate")
