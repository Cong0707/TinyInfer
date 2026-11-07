
import sys, time, queue
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.sched import ServingEngine
from tinyinfer.sequence import Sequence

class FakeEngine:
    """Minimal stand-in: forward returns logits whose argmax is the token count."""
    device = "cpu"
    pad_id = 0
    cfg = None
    def __init__(self):
        self.calls = 0
        self.shapes = []
    def forward(self, ids, start_pos=0, chunk=None, positions=None):
        import torch
        self.calls += 1
        self.shapes.append(tuple(ids.shape))
        b, t = ids.shape
        # logits [b, t, V]; argmax picks a token derived from the row index
        out = torch.zeros(b, t, 10)
        for i in range(b):
            out[i, :, (i + self.calls) % 10] = 1.0
        return out

eng = FakeEngine()
loop = ServingEngine(eng, max_batch=3)
print("max_batch:", loop.max_batch, "(cfg None -> 保持请求值)")
seqs = [Sequence(prompt=[1,2], max_new_tokens=3, temperature=0.0) for _ in range(3)]
for s in seqs: loop.submit(s)
t0 = time.time()
loop.start()
for s in seqs:
    got = []
    while True:
        it = s.queue.get(timeout=5)
        if it is None or isinstance(it, BaseException): break
        got.append(it)
    print(f"seq{s.id}: {got}")
loop.stop()
print("steps:", loop.steps, " tokens:", loop.tokens, f" {time.time()-t0:.2f}s")
print("forward 调用次数:", eng.calls, " shapes:", eng.shapes[:5])
print("同批合并:", any(s[0] > 1 for s in eng.shapes))
