"""按容量覆盖的显存缓存：反量化结果、KV 这类「可以重算但很贵」的东西。

专家的槽位池解决的是「权重在盘上、用到才搬到卡上」；这里的对象不一样 ——
它们的来源就在卡上（fp8 权重），缺了可以**重算**，只是重算很贵
（wo_a 一次反量化 = 64MB 的读 + 一次 fp32 中间张量）。

所以同一个思路照样成立：不专门去释放，按容量覆盖。满了就丢掉最久没用的，
下次用到再算一遍。小卡内存少就转得快，大卡一次装下就再也不重算 ——
**规模由容量自己决定，不是写死的层数**。
"""

from __future__ import annotations

import threading


class DerivedCache:
    """容量驱动的 LRU，键是 (layer, name)。

    和 expert 的槽位不同，这里没有「搬运」—— 值由 factory() 现算，
    所以 evict 只是丢引用，不涉及任何 IO。这也意味着覆盖永远是安全的：
    丢掉 = 下次重算，不会读错别人的数据。
    """

    def __init__(self, capacity_bytes: int = 0):
        self.capacity = max(0, int(capacity_bytes))
        self._items: dict[tuple[int, str], object] = {}
        self._order: list[tuple[int, str]] = []
        self._sizes: dict[tuple[int, str], int] = {}
        self._used = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _nbytes(self, v) -> int:
        try:
            return int(v.numel() * v.element_size())
        except Exception:
            return 0

    def get(self, layer: int, name: str):
        key = (int(layer), str(name))
        with self._lock:
            if key in self._items:
                self.hits += 1
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
                self._order.append(key)
                return self._items[key]
            self.misses += 1
            return None

    def put(self, layer: int, name: str, value):
        key = (int(layer), str(name))
        n = self._nbytes(value)
        with self._lock:
            old = self._items.get(key)
            if old is not None:
                self._used -= self._sizes.get(key, 0)
            self._items[key] = value
            self._sizes[key] = n
            self._used += n
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            # 按容量覆盖：从最久没用的开始丢，直到装得下
            while self._used > self.capacity and len(self._order) > 1:
                victim = self._order[0]
                if victim == key:
                    break
                self._order.pop(0)
                self._used -= self._sizes.pop(victim, 0)
                if self._items.pop(victim, None) is not None:
                    self.evictions += 1
        return value

    def get_or_compute(self, layer: int, name: str, factory):
        v = self.get(layer, name)
        if v is not None:
            return v
        v = factory()
        if v is not None:
            self.put(layer, name, v)
        return v

    def drop_layer(self, layer: int) -> int:
        n = 0
        with self._lock:
            for key in [k for k in self._items if k[0] == int(layer)]:
                self._used -= self._sizes.pop(key, 0)
                self._items.pop(key, None)
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
                n += 1
        return n

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._order.clear()
            self._sizes.clear()
            self._used = 0

    def stats(self) -> dict:
        return {"entries": len(self._items), "used_mb": self._used / 2**20,
                "capacity_mb": self.capacity / 2**20, "hits": self.hits,
                "misses": self.misses, "evictions": self.evictions}
