import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py"
t = io.open(p, encoding="utf-8").read()
old = """        eos = self._tok.eos_token_id
        dec = GraphDecoder.from_prompt(self._model, ids).capture()"""
new = """        eos = self._tok.eos_token_id
        dec, reused = self._decoder_for(ids)"""
assert old in t
t = t.replace(old, new, 1)

anchor = "    def health(self) -> dict[str, Any]:\n        return {\n            \"backend\": \"QwenBackend\","
helper = '''    def _decoder_for(self, ids):
        """Reuse a captured decoder when the prompt has the same length.

        A capture costs ~137ms: it runs the step several times to prove the replay
        matches eager. Rebuilding it for every request would put that on top of every
        completion, which is where the 174 tok/s of the decode loop drops to tens. The
        graph is position-shaped, not content-shaped, so identical prompt length means an
        identical graph; only the cache contents change, and those are reset per call.
        """
        from tinyinfer.hf_graph import GraphDecoder

        key = int(ids.shape[1])
        cached = self._graph_cache.get(key)
        if cached is not None:
            dec, cache = cached
            from transformers.cache_utils import StaticCache
            cfg = self._model.config.get_text_config()
            fresh = StaticCache(config=cfg, max_cache_len=dec.pos_buf[0].item() + 512
                                if False else dec.capacity)
            dec.cache = fresh
            dec.pos_buf.fill_(key)
            return dec, True
        dec = GraphDecoder.from_prompt(self._model, ids).capture()
        self._graph_cache[key] = (dec, None)
        return dec, False

'''
i = t.index(anchor)
t = t[:i] + helper + t[i:]
t = t.replace("        self._slots: dict[int, Any] = {}",
              "        self._slots: dict[int, Any] = {}\n        self._graph_cache: dict[int, Any] = {}", 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("patched")
