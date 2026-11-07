import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py"
t = io.open(p, encoding="utf-8").read()
old = """        length = int(ids.shape[1])
        dec = self._graphs.get(length)
        if dec is None:
            dec = GraphDecoder.from_prompt(self._model, ids).capture()
            if dec.using_graph:
                self._graphs[length] = dec
            else:
                # no graph for this shape: keep the decoder out of the cache so the next
                # request does not inherit a half-advanced state
                return dec
        else:
            dec.rewind()
        return dec, True"""
new = """        length = int(ids.shape[1])
        cached = self._graphs.get(length)
        if cached is not None:
            # rewind to the exact state replay expects: the capture advanced the cache by
            # one step, and this decoder has been replayed since
            cached.rewind()
            return cached, False
        dec = GraphDecoder.from_prompt(self._model, ids).capture()
        if dec.using_graph:
            self._graphs[length] = dec
        return dec, True"""
assert old in t, "body changed"
t = t.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("ok")