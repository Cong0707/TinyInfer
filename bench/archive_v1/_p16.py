import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py"
t = io.open(p, encoding="utf-8").read()
# The prep tokens belong to the prompt, not to the request that happened to trigger the
# capture: they are what the model emits right after prefill, so every request must emit
# them. Keep them with the cached decoder and hand them out on reuse too.
old = "        cached = self._graphs.get(length)\n        if cached is not None:"
new = """        cached = self._graphs.get(length)
        if cached is not None:
            # the prep tokens belong to the prompt, not to whichever request happened to
            # trigger the capture: they are what the model emits right after prefill
            self._prep.setdefault(length, getattr(cached, "prep_tokens", []))"""
assert old in t, "reuse branch missing"
t = t.replace(old, new, 1)

old2 = "        if dec.using_graph:\n            self._graphs[length] = dec\n        return dec, True"
new2 = """        if dec.using_graph:
            dec.prep_tokens = dec.drain()
            self._graphs[length] = dec
            self._prep[length] = dec.prep_tokens
            return dec, True
        return dec, True"""
assert old2 in t, "fresh branch missing"
t = t.replace(old2, new2, 1)

t = t.replace("        self._graphs: dict[int, Any] = {}",
              "        self._graphs: dict[int, Any] = {}\n        self._prep: dict[int, list[int]] = {}", 1)

old3 = "            for tok_id in (dec.drain() if fresh else ()):"
new3 = "            for tok_id in (dec.drain() if fresh else self._prep.get(length_of(ids), ())):"
assert old3 in t
t = t.replace(old3, new3, 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("ok")