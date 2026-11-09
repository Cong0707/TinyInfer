"""配置系统：一个文件、一套字段、一个读写入口。

## 为什么要把它们合并

之前配置散在三处：

  tuned.json        训练器扫出来的（按机器指纹分组）
  calibration.json  启动期现场实测的（按模型目录分组）
  os.environ        TI_SLOTS / TI_POOL_RATIO 等几十个环境变量

三处的键名、分组方式、默认值都不同，而且互相不知道对方存在。结果是：训练器扫出
`chosen.layout.slots`，规划器读不到；规划器算出来的值，运行时自适应又不知道；
运行时调好的值，下次启动没人写回文件。**每一环都要人手接一次，漏一环就静默失效。**

## 一个配置该长什么样

```
{
  "version": 2,
  "hardware": { ...指纹... },         # 硬件变了就整套作废
  "models": {
    "<模型指纹>": {
      "params": { "expert_slots": 8, ... },   # 生效中的参数
      "source": "formula|measured|runtime",   # 这个值是怎么来的
      "history": { "<参数>": [ ... ] }        # 每次变更的记录，可回归对比
    }
  }
}
```

`source` 是关键：同样是 `expert_slots=8`，来自「公式估算」和来自「实测确认」
的置信度完全不同。规划器可以按 source 决定要不要再测一遍。

## 生命周期

```
  启动    读配置 -> 有就用 -> 没有就用公式生成默认值并写回
  运行中  动态调整 -> 更新内存里的参数 -> 落盘（节流，不是每步都写）
  下次启动 直接读，除非硬件指纹或模型指纹变了
```

## 硬件变动

插了新显卡、内存变了、换机器 —— `hardware` 指纹对不上，整套作废重来。
这是唯一会让 `params` 失效的东西；模型换了只失效该模型那一项。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

_VERSION = 2
_lock = threading.RLock()
_cache = None
_dirty = False
_last_write = 0.0
_MIN_WRITE_INTERVAL = 5.0   # 运行时调优可能每步都想写，节流到 5 秒一次


def _home() -> Path:
    d = Path(os.environ.get("TI_CONFIG_DIR") or (Path.home() / ".dsh" / "plugins-data" / "tinyinfer"))
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def config_path() -> Path:
    return _home() / "config.json"


def hardware_key() -> str:
    """硬件指纹。插了新卡、内存变了，这个字符串就变。"""
    bits = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                bits.append("%s:%.0fG" % (p.name.replace(" ", "_"), p.total_memory / 2**30))
    except Exception:
        pass
    try:
        import psutil

        bits.append("ram=%.0fG" % (psutil.virtual_memory().total / 2**30))
    except Exception:
        pass
    bits.append(sys.platform)
    return "|".join(bits) if bits else "unknown"


def model_key(model_dir) -> str:
    return Path(str(model_dir)).name


def _empty() -> dict:
    return {"version": _VERSION, "hardware": hardware_key(), "models": {}}


def load() -> dict:
    """读配置。硬件指纹变了就返回空配置（旧的整套作废）。"""
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        data = None
        try:
            with open(config_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
        hw = hardware_key()
        if not isinstance(data, dict) or data.get("version") != _VERSION:
            data = _empty()
        elif data.get("hardware") != hw:
            # 硬件变了：参数全部作废，但把旧记录留在 .stale 里方便对照
            try:
                p = config_path()
                if p.exists():
                    p.replace(p.with_suffix(".stale.json"))
            except Exception:
                pass
            data = _empty()
        data.setdefault("models", {})
        _cache = data
        return data


def save(force: bool = False) -> None:
    """写回配置。运行时会频繁调，默认按 _MIN_WRITE_INTERVAL 节流。"""
    global _dirty, _last_write
    with _lock:
        if not _dirty and not force:
            return
        now = time.time()
        if not force and (now - _last_write) < _MIN_WRITE_INTERVAL:
            return
        data = _cache or _empty()
        data["version"] = _VERSION
        data["hardware"] = hardware_key()
        try:
            p = config_path()
            tmp = p.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            tmp.replace(p)     # 原子替换：写一半断电不会留下坏文件
            _last_write = now
            _dirty = False
        except Exception:
            pass


def get_model(model_dir) -> dict:
    return load().setdefault("models", {}).setdefault(model_key(model_dir), {})


def params(model_dir) -> dict:
    """该模型当前生效的参数。没有就返回空 dict（调用方自己回退到公式）。"""
    return dict(get_model(model_dir).get("params") or {})


def update(model_dir, values: dict, *, source: str = "runtime") -> None:
    """写入若干参数。source 说明这些值的来源，供下次启动判断可信度。

    source 取值：
      formula  规划器用经验公式算的（第一次启动）
      measured 训练器/启动期实测出来的
      runtime  运行时自适应调整过的（最贴合当下负载）
    """
    global _dirty
    if not values:
        return
    with _lock:
        ent = get_model(model_dir)
        p = ent.setdefault("params", {})
        hist = ent.setdefault("history", {})
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        changed = False
        for k, v in values.items():
            if p.get(k) == v:
                continue
            p[k] = v
            changed = True
            trail = hist.setdefault(k, [])
            trail.append({"ts": now, "value": v, "source": source})
            del trail[:-20]      # 只留最近 20 次，避免文件无限长
        if changed:
            ent["source"] = source
            ent["ts"] = now
            _dirty = True
    if changed:
        save()


def snapshot() -> dict:
    """当前配置的完整快照，给观测/排查用。"""
    return json.loads(json.dumps(load(), ensure_ascii=False))


def reset(model_dir=None) -> None:
    """丢弃参数。不传 model_dir 就整套重来。"""
    global _cache, _dirty
    with _lock:
        if model_dir is None:
            _cache = _empty()
        else:
            (load().get("models") or {}).pop(model_key(model_dir), None)
        _dirty = True
    save(force=True)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="查看/清理 tinyinfer 配置")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--path", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--model")
    a = ap.parse_args(argv)
    if a.path:
        print(config_path())
        return 0
    if a.reset:
        reset(a.model)
        print("reset ok")
        return 0
    print(json.dumps(snapshot(), ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
