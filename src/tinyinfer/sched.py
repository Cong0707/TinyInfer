"""Continuous batching: the serving loop that decodes several sequences at once.

The engine's own stream_generate() runs one sequence at a time. That throws away the
defining property of decode: a step reads every weight it touches exactly once, whether
it produces one token or eight. Eight sequences decoded together therefore cost one
weight read and produce eight tokens. On this checkpoint the read is ~12.7 GB per step,
43 ms of pure bandwidth at 295 GB/s; batching amortises that.

One class only. There were two (a slot-less prototype and this one) and keeping both
would mean two behaviours to maintain and two sets of bugs to find; the prototype is
gone and nothing used it.

What the loop does:

  admit    a request joins the running batch at a step boundary, without waiting for the
           current batch to drain -- static batching is what makes mixed load slow
  mix      prefill and decode share one forward call, because the weights for both are
           already resident; splitting them is two weight reads, i.e. the cost this
           module exists to remove
  retire   a finished sequence frees its KV slot for the next request immediately
  isolate  every sequence owns its position counter and stop set, so one finishing early
           never moves another's logits

The forward pass is the engine's own: batching is a property of the input tensor, not a
second implementation. It is NOT bit-identical to running the sequence alone -- cuBLAS
picks a different GEMM kernel for M=1 than for M>1 (measured: identical inputs, bf16
outputs one ulp apart), so any batched engine sees the same. The right acceptance test is
agreement of the sampled token, not equality of the logits.

"""

from __future__ import annotations

import time

import torch


class Engine:
    """The serving loop: admit, batch, forward once, sample, retire.

    Differs from BatchScheduler above in one respect that matters for a real server:
    it owns a KVPool, so a sequence that finishes frees its slot for the next request
    instead of the batch growing without bound. Admission is continuous -- a request
    joins the running batch at the next step boundary, not at the end of the current
    batch -- because waiting for a full batch to drain is what makes static batching
    slow under mixed load.
    """

    def __init__(self, engine, max_batch: int = 8):
        from .sequence import KVPool

        self.engine = engine
        self.max_batch = max(1, int(max_batch))
        self.pool = KVPool(self.max_batch)
        self.pending: list = []
        self.running: list = []
        self._fresh: set[int] = set()
        self.steps = 0
        self.tokens = 0
        self._step_ms = 0.0

    # ---- admission ----
    def submit(self, seq) -> None:
        self.pending.append(seq)

    def _admit(self) -> None:
        while self.pending:
            seq = self.pending[0]
            if self.pool.free <= 0:
                # 槽位满了。腾一个给**别的**会话用 —— 绝不腾给队首自己
                # （否则 park 完立刻又把它取出，无限循环）。
                if not self._make_room(exclude=seq.id):
                    break
            self.pending.pop(0)
            if self.pool.acquire(seq) is None:
                self.pending.insert(0, seq)
                break
            self.running.append(seq)
            self._fresh.add(seq.id)

    def _make_room(self, exclude: int | None = None) -> bool:
        """把一个「还在跑但很久没进展」的会话换出，给新请求让位。

        换出的是整条会话的 KV（KVPool.park 只保留历史，不切段），所以它
        回来时不用重新 prefill —— 1M 上下文重新 prefill 是几分钟。
        返回是否真的腾出了一个槽位。
        """
        candidates = [s for s in self.running if s.id != exclude]
        if not candidates:
            return False
        # 挑最早提交的（最可能已经跑完大部分），换出去最划算
        victim = min(candidates, key=lambda s: (s.n_out, s.id))
        try:
            nbytes = self.pool.park(victim)
        except Exception:
            return False
        self.running = [s for s in self.running if s.id != victim.id]
        self._fresh.discard(victim.id)
        # 它没跑完，放回等待队列（历史已经在 pool 里）
        victim.done = False
        self.pending.append(victim)
        return nbytes >= 0

    def _retire(self) -> list:
        out = []
        keep = []
        for seq in self.running:
            finished = seq.done or seq.n_out >= seq.max_new_tokens
            if finished:
                self.pool.release(seq)
                out.append(seq)
            else:
                keep.append(seq)
        self.running = keep
        return out

    # ---- one step ----
    def _width(self) -> int:
        fresh = [s for s in self.running if s.id in self._fresh]
        return max((len(s.prompt) for s in fresh), default=1)

    def step(self) -> list:
        """Advance every running sequence one token; returns the finished ones."""
        from .nn import sample_logits

        self._admit()
        if not self.running:
            return []
        width = self._width()
        pad = self.engine.pad_id
        ids = torch.full((len(self.running), width), pad, dtype=torch.long,
                         device=self.engine.device)
        for i, s in enumerate(self.running):
            if s.id in self._fresh:
                ids[i, :len(s.prompt)] = torch.tensor(s.prompt, dtype=torch.long)
            else:
                ids[i, 0] = int(s.out[-1])
        # per-row positions: a fresh prompt's last real column is len(prompt)-1, a
        # decoding row's only column is its own next position
        pos = torch.tensor(
            [(len(s.prompt) - 1) if s.id in self._fresh else s.next_pos
             for s in self.running],
            dtype=torch.long, device=self.engine.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self.engine.forward(ids, start_pos=int(pos.min().item()),
                                         positions=pos)
        self._step_ms = (time.perf_counter() - t0) * 1000.0
        self.steps += 1

        finished = []
        for i, s in enumerate(self.running):
            col = (len(s.prompt) - 1) if s.id in self._fresh else 0
            nid = sample_logits(logits[i, col], s.temperature, s.top_p, s.top_k)
            s.out.append(int(nid))
            self.tokens += 1
            self._fresh.discard(s.id)
            if s.stop_ids and nid in s.stop_ids:
                s.done = True
        return finished + self._retire()

    def run(self, budget_steps: int = 100000) -> list:
        """Serve until nothing is pending or running; returns every sequence."""
        served = list(self.pending) + list(self.running)
        for _ in range(budget_steps):
            self.step()
            if not self.running and not self.pending:
                break
        return served

    def stats(self) -> dict:
        d = {
            'steps': self.steps,
            'tokens': self.tokens,
            'running': len(self.running),
            'pending': len(self.pending),
            'last_step_ms': round(self._step_ms, 2),
        }
        d.update(self.pool.stats())
        return d

class ServingEngine:
    """The always-on loop: one thread steps the batch, requests come and go.

    This is the shape a server needs, and it is the default rather than a mode. The
    caller submits a Sequence and drains a queue of token ids; nothing in that interface
    mentions batching, so a single concurrent request and eight of them take the same
    path. The KVPool decides how many can actually share a step -- when it only has one
    slot (a card that cannot hold two sequences' KV) the loop still runs, one sequence at
    a time, and the API behaves exactly as it did before.

    Why a thread and not an async loop: every step blocks on the GPU, and the engine is
    one mutex-guarded object with a single CUDA context. A thread that owns the device
    and a queue in front of it is the simplest thing that cannot interleave two forwards.
    """

    def __init__(self, engine, max_batch: int = 8, idle_sleep: float = 0.002):
        import queue
        import threading

        self._queue_mod = queue
        self.engine = engine
        self.max_batch = max(1, int(max_batch))
        self.idle_sleep = float(idle_sleep)
        self._inbox: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.steps = 0
        self.tokens = 0
        self._running: list = []
        self._fresh: set[int] = set()
        # 槽位池：决定「这一步能装几条会话」，也是 park/resume 的落点。
        # 文档里一直写着「KVPool decides how many can share a step」，但类里
        # 从来没有它 —— 只有「运行数 >= max_batch」的条数上限。于是
        # 「槽位不够时换出旧会话」这条策略根本没生效过。
        from .sequence import KVPool

        # 换出池的字节上限：按真实可用内存的一半算，不写死。
        # 一条 1M 会话的压缩 KV 是 6.9GB，光按条数限（默认 64）会要 440GB。
        _park_bytes = 0
        try:
            import psutil as _ps

            _park_bytes = int(_ps.virtual_memory().available * 0.5)
        except Exception:
            _park_bytes = 0
        self.pool = KVPool(self.max_batch, max_parked_bytes=_park_bytes)
        # 被换出、等槽位的会话。不放进 _inbox：放进去的话本次 _admit 的 while
        # 会立刻又把它取出来、又换出去，无限循环（实测直接超时）。
        self._parked_wait: list = []
        # What the device can actually hold, not what the caller asked for. A sequence
        # needs its window ring plus its compressed groups in VRAM for the whole life of
        # the batch (they cannot be swapped out per step without defeating the point), and
        # the model itself is already resident. Asking for 8 on a 7 GB card would OOM on
        # the second sequence, so the ceiling is measured and the request is clamped
        # rather than trusted. This runs last: it reads the fields above.
        # 以前这里被迫把 max_batch 压到 1：多序列拼批会让 batch 维从 1 变 2，
        # 而 attention 里按行算窗口的那段把 positions([b, s]) 当标量用，
        #   TypeError: int() argument must be ... not 'list'
        # 于是每次并发请求都有 20-30% 概率 500。那个 bug 已经修掉
        # （tests/test_batched_forward.py 锁住 batch=2 的前向），所以这里的
        # 上限只由 _fit() 按真实显存算出的数字决定，不再人为压到 1。
        # TI_MAX_BATCH 仍然可以覆盖（压测要能固定住）。
        _mb_env = __import__("os").environ.get("TI_MAX_BATCH")
        if _mb_env is not None:
            self.max_batch = max(1, int(_mb_env))
        self.max_batch = min(self.max_batch, self._fit())

    def _fit(self) -> int:
        """How many sequences the free VRAM can hold at the same time.

        Sized from the attention geometry, not from a constant: the window ring is
        sliding_window x head_dim x 2 bytes per layer, and the compressed cache is one
        group per `ratio` tokens. A 4K context is the working assumption because a
        shorter one only makes the estimate conservative.
        """
        try:
            cfg = getattr(self.engine, "cfg", None)
            if cfg is None:
                return self.max_batch
            try:
                import torch as _t

                if not _t.cuda.is_available():
                    return self.max_batch
                free, _total = _t.cuda.mem_get_info()
            except Exception:
                return self.max_batch
            # keep a quarter for activations, the head and the allocator
            budget = int(free * 0.25)
            win = int(getattr(cfg, "sliding_window", 0) or 0)
            hd = int(getattr(cfg, "head_dim", 0) or 0)
            nl = int(getattr(cfg, "n_layers", 0) or 0)
            if win <= 0 or hd <= 0 or nl <= 0:
                return self.max_batch
            per_seq = win * hd * 2 * nl
            ctx = 4096
            for r in list(getattr(cfg, "compress_ratios", []) or []):
                if r and r > 0:
                    per_seq += (ctx // int(r)) * hd * 2
            if per_seq <= 0:
                return self.max_batch
            return max(1, min(self.max_batch, budget // per_seq))
        except Exception:
            return 1

    # ---- lifecycle ----
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = __import__("threading").Thread(
            target=self._loop, name="tinyinfer-serve", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=10.0)
        self._thread = None

    # ---- submission ----
    def submit(self, seq):
        """Queue a sequence; it is admitted at the next step boundary."""
        import queue as _q

        seq.queue = _q.Queue()
        self._inbox.put(seq)
        return seq

    def _admit(self) -> None:
        # 上一轮换出去的会话再试一次（槽位可能已经空出来了）
        while self._parked_wait:
            self._inbox.put(self._parked_wait.pop(0))
        while True:
            try:
                seq = self._inbox.get_nowait()
            except self._queue_mod.Empty:
                break
            # 槽位满时先试着换出一个旧会话（KV 沉到内存），而不是让新请求干等。
            # 以前这里只有条数判断：running 满就塞回队列，队列里所有请求要等
            # 正在跑的全部结束 —— 对「几个长会话轮流推进」的服务来说等于卡死。
            if self.pool.free <= 0:
                if not self._make_room(exclude=getattr(seq, "id", None)):
                    self._inbox.put(seq)
                    break
            slot = self.pool.acquire(seq)
            if slot is None:
                self._inbox.put(seq)
                break
            self._running.append(seq)
            # 只有「从没跑过」的会话才是 fresh（要送整个 prompt 做 prefill）。
            # 被 park 换出又回来的会话已经产过 token，必须按 decode 送最后那个 ——
            # 否则它会从头重放 prompt，out 里已有的 token 被忽略（实测单槽位下
            # 第二条会话反复拿到同一个 token：[2, 2, 2]）。
            if not seq.out:
                self._fresh.add(seq.id)

    def _make_room(self, exclude: int | None = None) -> bool:
        """换出一个还在跑但进展最少的会话，腾出槽位（KV 留在 pool 里）。

        被换出的会话进 `_parked_wait`，不回到 `_inbox` —— 回 inbox 的话
        本次 _admit 的 while 会立刻把它取出来，又换出，无限循环（实测超时）。
        `_admit` 开头先把 `_parked_wait` 挪回 inbox，轮到它时自然会拿到槽位。
        """
        candidates = [s for s in self._running
                      if getattr(s, "id", None) != exclude]
        if not candidates:
            return False
        victim = min(candidates, key=lambda s: (s.n_out, s.id))
        try:
            self.pool.park(victim)
        except Exception:
            return False
        self._running = [s for s in self._running if s.id != victim.id]
        self._fresh.discard(victim.id)
        self._parked_wait.append(victim)
        return True

    # ---- the loop ----
    def _loop(self):
        while not self._stop.is_set():
            try:
                did = self.step()
            except Exception as exc:  # a failed step must not kill the server thread
                for seq in list(self._running):
                    _fail(seq, exc)
                self._running.clear()
                self._fresh.clear()
                did = False
            if not did:
                self._stop.wait(self.idle_sleep)

    def _width(self) -> int:
        fresh = [s for s in self._running if s.id in self._fresh]
        return max((len(s.prompt) for s in fresh), default=1)

    def step(self) -> bool:
        """One forward over the whole running batch. False when there was nothing to do."""
        from .nn import sample_logits

        self._admit()
        if not self._running:
            return False
        # prefill 和 decode 不能混在一个 batch 里：fresh 序列的 ids 是
        # [B, prompt_len] 而解码中的序列只有一列，按最宽的那个开 buffer 后
        # positions 各不相同，某些层会按 start_pos 统一处理，于是 batch 维在中间
        # 某一步从 1 变成 2，抛
        #   output with shape [1, T, 512] doesn't match [2, T, 512]
        # （实测 10 次请求 2-3 次 500）。有 fresh 时就只推进它自己，等其他序列
        # 都进入 decode（每步一律 1 个 token、宽度相同）再合并成真正的 batch。
        width = self._width()
        pad = self.engine.pad_id
        ids = torch.full((len(self._running), width), pad, dtype=torch.long,
                         device=self.engine.device)
        for i, s in enumerate(self._running):
            if s.id in self._fresh:
                ids[i, :len(s.prompt)] = torch.tensor(s.prompt, dtype=torch.long)
            else:
                ids[i, 0] = int(s.out[-1])
        pos = torch.tensor(
            [(len(s.prompt) - 1) if s.id in self._fresh else s.next_pos
             for s in self._running],
            dtype=torch.long, device=self.engine.device)

        with torch.inference_mode():
            logits = self.engine.forward(ids, start_pos=int(pos.min().item()),
                                         positions=pos)
        self.steps += 1

        done = []
        for i, s in enumerate(self._running):
            col = (len(s.prompt) - 1) if s.id in self._fresh else 0
            nid = sample_logits(logits[i, col], s.temperature, s.top_p, s.top_k)
            s.out.append(int(nid))
            self.tokens += 1
            self._fresh.discard(s.id)
            q = getattr(s, "queue", None)
            if q is not None:
                q.put(int(nid))
            if (s.stop_ids and nid in s.stop_ids) or s.n_out >= s.max_new_tokens:
                s.done = True
                done.append(s)
        for s in done:
            q = getattr(s, "queue", None)
            if q is not None:
                q.put(None)          # end marker
            # 会话跑完：槽位还回池子，下一步就能接新请求。
            # 以前没有这一句（那时类里根本没有 pool）。
            try:
                self.pool.release(s)
            except Exception:
                pass
        self._running = [s for s in self._running if not s.done]
        return True

    def stats(self) -> dict:
        return {"steps": self.steps, "tokens": self.tokens,
                "running": len(self._running), "max_batch": self.max_batch}


def _fail(seq, exc) -> None:
    q = getattr(seq, "queue", None)
    if q is not None:
        q.put(exc)
