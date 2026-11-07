import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py"
t = io.open(p, encoding="utf-8").read()

# The reused decoder must not drain: drain() reports tokens produced while *capturing*,
# which only exist for the request that did the capture. After rewind() the state is
# exactly "ready for the next replay", so the first generated token comes from replay.
old = "            for tok_id in dec.drain():"
assert old in t
t = t.replace(old,
              "            for tok_id in (dec.drain() if fresh else ()):", 1)

old2 = "        dec = self._decoder_for(request, ids)"
new2 = """        dec, fresh = self._decoder_for(request, ids)"""
assert old2 in t
t = t.replace(old2, new2, 1)

old3 = "        return dec\n\n    def health"
new3 = "        return dec, True\n\n    def health"
if old3 in t:
    t = t.replace(old3, new3, 1)

io.open(p, "w", encoding="utf-8", newline="").write(t)
print("patched")