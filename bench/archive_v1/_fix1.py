import io
p = r"E:\Develop\Deepseek\tinyinfer\tests\test_server.py"
t = io.open(p, encoding="utf-8").read()
old = """    class FakeEngine:
        def __init__(self):
            self.kv = {}
            self.resets = 0"""
new = """    class FakeEngine:
        # _bind_slot moves restored tensors to self._model.device
        device = "cpu"

        def __init__(self):
            self.kv = {}
            self.resets = 0"""
assert old in t
t = t.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("ok")