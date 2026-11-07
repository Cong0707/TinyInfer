import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\qwen_model.py"
t = io.open(p, encoding="utf-8").read()

# checkpoint naming + kv factory, per the protocol
old = '    family = "qwen3_5"'
new = """    family = "qwen3_5"
    # this checkpoint nests everything under model.language_model and ties the output
    # head to the embedding
    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    head_key = None"""
assert old in t
t = t.replace(old, new, 1)

old2 = """    def forward(self, ids, start_pos: int, engine) -> torch.Tensor:
        \"\"\"Delegates to the reference transformer body, layer by layer via the engine.\"\"\"
        from . import qwen_kernels as K
        return K.model_forward(self, engine, ids, start_pos)\"""
new2 = """    def make_kv(self, capacity: int):
        from .kv import QwenKv
        return QwenKv(self.cfg, capacity)

    def forward(self, ids, start_pos: int, engine) -> torch.Tensor:
        \"\"\"Delegates to the reference transformer body, layer by layer via the engine.\"\"\"
        from . import qwen_kernels as K
        return K.model_forward(self, engine, ids, start_pos)"""
assert old2 in t
t = t.replace(old2, new2, 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("qwen_model extended")