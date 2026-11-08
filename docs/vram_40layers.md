# 40 层全模型在 8GB 上仍然起不来（未解决）

## 现象

`probe_chunk40.py`（40 层、真实 checkpoint）：

```
chunk= 4 FAIL MoE layer 0: rows 0-3 unavailable [line 375]
chunk= 8 FAIL !handles_.at(i) INTERNAL ASSERT FAILED
chunk=16 FAIL !handles_.at(i) INTERNAL ASSERT FAILED
```

`[arena] layer 0: slots=8 ... VRAM expert slot arena allocation failed`

## 账对不上（已经查清的部分）

规划器（`resources.plan`）对 8GB 卡给出：

```
layers=11/40  stream=29  buf=3  slots=11  chunk=8
```

按规划器的公式，11 层 + 11 槽位应该占 **4.15GB**：

```
11 x 164MB x LAYER_SLACK(1.70)  = 3.07GB
3 x 177.6MB (buffers)           = 0.52GB
ws0 (0.20 + 8x0.02)             = 0.36GB
11 x 18.8MB (slots)             = 0.20GB
                                  ------
                                  4.15GB
```

实测初始化后 **alloc = 6.18GB，free = 0.36GB**。差 **2.03GB**。

## 已经排除的

- **MTP**：`mtp_bytes = 1.33GB`。日志明确说「MTP off: the draft stages do not fit」，
  而且 `speculation` 在 `V < 16GB` 时被关掉，引擎不分配。试着无条件计入反而把
  slots 从 11 压到 1，是错的，已回退。
- **engram**：`engram_bytes = 0.59GB`，`runtime_b` 里已含 `0.5x = 0.29GB`。

## 还没查清的 2GB

最可能的方向：

1. **层权重大小的口径不一致**。规划器用 `layer_bytes=164MB`（不含 expert），
   再乘 `LAYER_SLACK=1.70` 得到 279MB/层。但引擎常驻一层时实际分配的是
   `max_layer_bytes=177.6MB` 的 bf16 权重 + 反量化缓存（wo_a 等），实测注释里
   写过「355MB 的 bf16 权重占约 600MB 显存」。如果真实是 600MB/层，
   11 层就是 6.6GB —— 和实测吻合。
2. **`wo_a` 反量化缓存**：`_derived()` / `_st["wo_a_bf"]` 每层留一份
   37MB（32768x1280 bf16），11 层 = 0.4GB。

## 结论

**40 层全模型在 8GB 卡上跑不起来**（4 层可以）。这是显存规划的真实缺陷，
`LAYER_SLACK=1.70` 可能低估了。

**但这不是目标场景**：用户要的是 L20（44GB）。L20 的规划是 `resident=40,
slots=1113, chunk=139`，40 层全常驻、1100+ 槽位 —— 那个配置下 8GB 的这
个问题不存在。

## 下一步

1. 精确测量「一层常驻真实占多少显存」（把 `LAYER_SLACK` 校准到实测值）。
2. 或者：当 `free < slots*blob + 200MB` 时，在引擎里主动把 slots 降到 0，
   走 `TI_NO_SLOTS` 路径 —— 至少让模型能跑，只是慢。

**在那之前，8GB 卡只能跑较少层数（4-11 层已验证），40 层不行。**
