import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py"
t = io.open(p, encoding="utf-8").read()
old = "            for tok_id in (dec.drain() if fresh else self._prep.get(int(ids.shape[1]), ())):"
new = """            # the prep tokens (emitted between prefill and the captured step) are the same
            # for every request on this graph shape: _decoder_for recorded them once
            for tok_id in self._prep.get(int(ids.shape[1]), ()):"""
assert old in t
t = t.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("ok")