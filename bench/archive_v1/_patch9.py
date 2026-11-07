import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\server.py"
t = io.open(p, encoding="utf-8").read()
old = """    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        from tinyinfer.hf_graph import GraphDecoder

        torch = self._torch"""
new = """    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        with self._torch.inference_mode():
            yield from self._generate(request)

    def _generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        from tinyinfer.hf_graph import GraphDecoder

        torch = self._torch"""
assert old in t
t = t.replace(old, new, 1)
old2 = """    def _decoder_for(self, request, ids):"""
new2 = """    def _decoder_for(self, request, ids):
        # called inside inference_mode (see generate): the snapshot restore and the
        # position bookkeeping both write inference tensors in place"""
assert old2 in t
t = t.replace(old2, new2, 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("patched")
