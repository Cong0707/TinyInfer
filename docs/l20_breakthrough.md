# L20 突破：装下全部专家后 prefill 快 24 倍、decode 快 3 倍（2026-11-08 16:46）

## 实测对比

| 指标 | 修复前 | 修复后 | 提升 |
| --- | --- | --- | --- |
| prefill (64 token) | 114.0s | **4.8s** | **24x** |
| decode 稳态 | 300-1900ms | **130-212ms** | **3-10x** |
| decode 读盘 | 13-25 次/步 | **0-1 次/步** | 消除 |
| 吞吐 | 0.85 tok/s | **4.5-7.7 tok/s** | **5-9x** |

## 根因链（三个叠加的限制）

1. **RAM 缓存装不下全部专家**
   - `resources.py`：`cache_b = (host - slab - tables - page) * 0.8`
   - `engine.py:436`：`_cap_bytes = _pool_avail * 0.75`（给未加载的层权重留 25%）
   - 两次打折后 L20 的池子只有 **231GB**，而模型全部 15360 个专家要 **268.9GB**
   - 差 14% → 预热永远装不完 → decode 每步去读盘

2. **25% 的余量在「全层常驻」时是白留的**
   - 那个余量是给 streamed 层权重留的，而 L20 上 40/40 全常驻，没有 streamed 层

3. **预热顺序是逐层填满**
   - `for layer in range(40)`：填到第 6 层时第 30 层还是空的，而 decode 每步要用全部 40 层
   - 改成按专家位置跨层轮转（每层先填 0-7 号，再 8-15 号）

## 修法

```python
# engine.py：全层常驻时只留 6% 余量
_n_stream = len(self._stream_layers())
_frac = 0.94 if _n_stream == 0 else 0.75
_cap_bytes = max(192 << 20, int(_pool_avail * _frac))
```

结果：`cap=16561 needed=15360 fits=True`，池子 290GB。

## 现在的水平

```
预热 210s（15139 个专家，76 个/秒）
prefill 4.8s
decode 稳态 130-212ms = 4.7-7.7 tok/s
```

理论预算（40 层 x 0.95ms MoE + attention + hc）约 64ms/step = 15.6 tok/s。
**现在达到理论的 30-50%。**

## 剩余差距在哪

`d_disk` 已经接近 0，说明读盘基本消除。剩下的 130ms/step 里：

- MoE 计算：40 x 0.95ms = 38ms
- attention + hc + 其它：约 26ms（理论）
- **未知开销：约 66ms**

下一步要看这 66ms 是什么（逐层计时，现在读盘噪音已经没了，数据会干净很多）。
