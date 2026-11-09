"""启动期标定：实测这台机器的真实带宽和最优并发深度。

规划器（resources.plan）只能从「硬件探测 + 模型头信息」推断，有些量它猜不出来，
因为它们不是硬件规格，而是**这台机器上这套驱动 + 这块盘 + 这个文件布局**的实测
性质：

  - 磁盘顺序读带宽（决定 slab 装载速度）
  - 磁盘随机读带宽（决定专家 miss 的代价 —— 专家 blob 在分片里是散的）
  - 随机读的最优并发深度（队列深度饱和点，跟 CPU 数没关系）
  - pinned H2D 带宽（决定每步层传输的代价）

这些量以前是写死的常数（16 线程、26.5GB/s、3.8GB/s），换一台机器就不对，
而且看不出来它错了 —— 只会表现为「引擎在那台机器上莫名慢」。

这里改成启动时实测一遍，结果缓存到磁盘（按机器指纹 + 模型路径做键），
下次启动直接读缓存，不再重复测。

**只测不猜**：任何一个量测不出来就返回 None，让调用方退回规划器的推断值，
而不是塞一个「差不多」的常数进去。
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path


def _cache_path() -> Path:
    base = os.environ.get("TI_CALIB_DIR")
    if base:
        p = Path(base)
    else:
        p = Path.home() / ".dsh" / "plugins-data" / "tinyinfer"
    return p / "calibration.json"


def _machine_key() -> str:
    """机器指纹：只取会影响带宽的硬指标，不取主机名之类无关的。"""
    bits = []
    try:
        import psutil

        bits.append(f"ram={int(psutil.virtual_memory().total // (1 << 30))}")
        bits.append(f"cpu={psutil.cpu_count(logical=True)}")
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            bits.append(f"gpu={torch.cuda.get_device_name(0)}")
            bits.append(f"vram={int(torch.cuda.get_device_properties(0).total_memory // (1 << 30))}")
    except Exception:
        pass
    bits.append(f"os={os.name}")
    return "|".join(bits)


def _read_bytes(path, offset: int, n: int) -> int:
    with open(path, "rb", buffering=0) as f:
        f.seek(offset)
        buf = bytearray(min(n, 4 << 20))
        got = 0
        while got < n:
            k = f.readinto(buf)
            if not k:
                break
            got += k
    return got


def measure_disk(model_dir: str | Path) -> dict:
    """测顺序读和随机读带宽。

    顺序：连续读一个分片的前若干 MB。
    随机：在分片里随机挑偏移读 8MB 块 —— 专家 blob 就是这种访问模式，
    两者的比值能说明「这块盘适不适合当前这种读取形态」。
    """
    out: dict = {"seq_gbps": None, "rand_gbps": None}
    # 直接列目录里最大的几个分片。不走 ShardIndex 的 shard_path() 接口 ——
    # 不同版本的分片命名方式不一样（有的按序号，有的按内容哈希），
    # 这里只需要「几个真实存在的大文件」，用文件系统说话最稳。
    try:
        cand = []
        for e in Path(str(model_dir)).iterdir():
            if e.suffix != ".safetensors" or not e.is_file():
                continue
            try:
                cand.append((e.stat().st_size, str(e)))
            except Exception:
                continue
        cand.sort(reverse=True)
        paths = [pth for _sz, pth in cand[:4]]
    except Exception:
        return out
    if not paths:
        return out
    random.seed(1234)
    try:
        p = paths[0]
        size = os.path.getsize(p)
        # 顺序：读 96MB
        n = min(96 << 20, max(8 << 20, size // 8))
        t0 = time.perf_counter()
        got = _read_bytes(p, 0, n)
        dt = time.perf_counter() - t0
        if dt > 0 and got > 0:
            out["seq_gbps"] = round(got / 1e9 / dt, 2)
        # 随机：8 个 8MB 块
        total, dt = 0, 0.0
        for _ in range(8):
            off = random.randrange(0, max(1, size - (8 << 20)))
            t0 = time.perf_counter()
            total += _read_bytes(p, off, 8 << 20)
            dt += time.perf_counter() - t0
        if dt > 0 and total > 0:
            out["rand_gbps"] = round(total / 1e9 / dt, 2)
    except Exception:
        pass
    return out


def tune_read_threads(model_dir: str | Path, index=None, *, candidates=(4, 8, 16, 32),
                      mb: int = 24) -> dict:
    """实测「随机读在几个并发下最快」。

    做法：随机挑一批专家 blob，用每个候选并发各读一遍同样的字节数，取吞吐最高
    的那个。这是唯一能给出这个数的方法 —— SSD 的队列深度饱和点取决于盘、
    驱动、以及文件在盘上的物理分布，跟 CPU 核数没有关系。

    返回 {"threads": 最优值或 None, "measured": {并发: GB/s}}
    """
    out: dict = {"threads": None, "measured": {}}
    try:
        from .loading import ShardIndex
        from concurrent.futures import ThreadPoolExecutor

        idx = index or ShardIndex(str(model_dir))
        blobs = []
        random.seed(4321)
        for _ in range(24):
            lay = random.randrange(0, max(1, int(getattr(idx, "n_layers", 40) or 40)))
            exp = random.randrange(0, 384)
            try:
                blobs.append(idx.expert_blob(lay, exp))
            except Exception:
                continue
        if len(blobs) < 4:
            return out
        # 只取每个 blob 的第一段，控制总字节数
        segs = []
        for b in blobs:
            if b.segments:
                segs.append(b.segments[0])
        target = max(1, min(len(segs), int(mb * (1 << 20) // max(1, segs[0].nbytes))))
        work = segs[: max(2, target)]

        def one(seg):
            buf = bytearray(seg.nbytes)
            mv = memoryview(buf)
            with open(str(idx.shard_path(seg.shard)), "rb", buffering=0) as f:
                f.seek(seg.file_offset)
                got = 0
                while got < seg.nbytes:
                    k = f.readinto(mv[got:])
                    if not k:
                        break
                    got += k
            return seg.nbytes

        for th in candidates:
            try:
                t0 = time.perf_counter()
                if th <= 1:
                    total = sum(one(s) for s in work)
                else:
                    with ThreadPoolExecutor(max_workers=int(th)) as ex:
                        total = sum(ex.map(one, work))
                dt = time.perf_counter() - t0
                if dt > 0 and total > 0:
                    out["measured"][int(th)] = round(total / 1e9 / dt, 2)
            except Exception:
                continue
        if out["measured"]:
            out["threads"] = max(out["measured"], key=lambda k: out["measured"][k])
    except Exception:
        pass
    return out


def load(model_dir: str | Path) -> dict | None:
    """读缓存的标定结果。"""
    try:
        p = _cache_path()
        if not p.is_file():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        key = f"{_machine_key()}|{Path(str(model_dir)).name}"
        return data.get(key)
    except Exception:
        return None


def save(model_dir: str | Path, result: dict) -> None:
    try:
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[f"{_machine_key()}|{Path(str(model_dir)).name}"] = result
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def calibrate(model_dir: str | Path, *, force: bool = False, quick: bool = True) -> dict:
    """跑一遍标定（有缓存就直接用）。"""
    if not force:
        cached = load(model_dir)
        if cached:
            cached["cached"] = True
            return cached
    res: dict = {"cached": False, "ts": time.time()}
    res.update(measure_disk(model_dir))
    if quick:
        tr = tune_read_threads(model_dir)
        res["read_threads"] = tr.get("threads")
        res["read_threads_measured"] = tr.get("measured")
    try:
        save(model_dir, res)
    except Exception:
        pass
    return res
