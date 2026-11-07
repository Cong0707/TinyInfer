import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py"
t = io.open(p, encoding="utf-8").read()

start = t.index("    # -- capture/KV reuse through .kvcache ---------------------------------")
end = t.index("    def health(self) -> dict[str, Any]:\n        return {\n            \"backend\": \"QwenBackend\",")
new = """    # -- capture reuse -----------------------------------------------------
    def _decoder_for(self, request, ids):
        \"\"\"A captured decoder for this prompt length.

        A capture costs ~137ms (it runs the step several times to prove the replay matches
        eager). Building one per request puts that on top of every completion, which is
        where the 174 tok/s of the raw decode loop collapses to tens. The graph depends on
        the cache *shape*, not its contents, so it is cached by prompt length; GraphDecoder
        .rewind() puts the state back to what replay expects before each use.
        \"\"\"
        from tinyinfer.hf_graph import GraphDecoder

        length = int(ids.shape[1])
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
        return dec

"""

t = t[:start] + new + t[end:]
# drop the _Snapshot helper: rewind() lives on the decoder now
h0 = t.find("class _Snapshot:")
if h0 >= 0:
    h1 = t.find("class QwenBackend(BaseBackend):")
    t = t[:h0] + t[h1:]
t = t.replace("        self._graphs: dict[int, Any] = {}\n        self._base: dict[int, Any] = {}",
              "        self._graphs: dict[int, Any] = {}", 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("server patched")