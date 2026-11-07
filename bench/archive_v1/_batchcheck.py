
import sys, torch, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.sched import ServingEngine
from tinyinfer.sequence import Sequence

# 用真引擎但只 1 层，快速验证合批
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
eng = Engine(r"E:\models\DeepSeek-V4.1-Flash-1017", tier=OptLevel.AGGRESSIVE,
             device="cuda", layers=[0])

shapes = []
orig = eng.forward
def spy(ids, start_pos=0, chunk=None, positions=None):
    shapes.append(tuple(ids.shape))
    return orig(ids, start_pos, chunk, positions=positions)
eng.forward = spy

loop = ServingEngine(eng, max_batch=4)
print("max_batch:", loop.max_batch)
seqs=[Sequence(prompt=[1,2,3], max_new_tokens=2, temperature=0.0) for _ in range(4)]
for s in seqs: loop.submit(s)
loop.start()
for s in seqs:
    while True:
        it = s.queue.get(timeout=120)
        if it is None or isinstance(it, BaseException): break
loop.stop()
print("forward shapes:", shapes)
print("合批:", any(s[0] > 1 for s in shapes), " 步数:", loop.steps)
