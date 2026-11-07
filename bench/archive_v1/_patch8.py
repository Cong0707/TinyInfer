import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py"
t = io.open(p, encoding="utf-8").read()
old = """    def restore(self, cache) -> None:
        from tinyinfer.hf_graph import _cache_restore
        _cache_restore(self.snap)
        # _cache_restore writes the tensors; the per-layer bookkeeping (cumulative_length,
        # written counters) has to be reset too or the mask comes out wrong
        for layer in getattr(cache, "layers", []):
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length.zero_()
            if hasattr(layer, "written"):
                layer.written = 0"""
new = """    def restore(self, cache) -> None:
        import torch as _t
        from tinyinfer.hf_graph import _cache_restore
        # the snapshot holds inference tensors, which refuse in-place writes outside
        # InferenceMode; the restore itself is a pure in-place copy
        with _t.inference_mode():
            _cache_restore(self.snap)
            # _cache_restore writes the tensors; the per-layer bookkeeping has to be reset
            # too or the mask comes out one position too long
            for layer in getattr(cache, "layers", []):
                if hasattr(layer, "cumulative_length"):
                    layer.cumulative_length.zero_()
                if hasattr(layer, "written"):
                    layer.written = 0"""
assert old in t
t = t.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("patched")
