import io
p = r"E:\Develop\Deepseek\tinyinfer\tests\test_server.py"
t = io.open(p, encoding="utf-8").read()
old = """    backend = TinyModelBackend(model_dir=None, tier="conservative")
    engine = FakeEngine()
    backend._model = engine
    backend._cfg = SimpleNamespace(max_position_embeddings=128)"""
new = """    backend = TinyModelBackend(model_dir=None, tier="conservative")
    engine = FakeEngine()
    backend._model = engine
    backend._cfg = SimpleNamespace(max_position_embeddings=128)
    # the slot helpers first look for .kvcache/slot3.pt on disk, so a machine that has
    # served a real request would make this test depend on leftover local state
    for leftover in (backend._kv_path("slot3"), backend._kv_path("slot4")):
        leftover.unlink(missing_ok=True)"""
assert old in t
t = t.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("ok")