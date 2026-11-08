import sys, time, threading
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
from tinyinfer.expert_queue import ExpertServer, ExpertRequestRing, ReadyFlag

class Mgr:
    stats = {}
    def need(self, layer, ids):
        raise AssertionError("no request")

# 只测 drain 的频率
ring = ExpertRequestRing()
ready = ReadyFlag()
srv = ExpertServer(Mgr(), ring, ready, poll_us=2000)
calls = {"n": 0}
_d = ring.drain
def d():
    calls["n"] += 1
    return _d()
ring.drain = d
srv.start()
t0 = time.process_time()
time.sleep(0.4)
used = time.process_time() - t0
srv.stop()
print("idle cpu=%.3fs  drain calls=%d" % (used, calls["n"]), flush=True)
