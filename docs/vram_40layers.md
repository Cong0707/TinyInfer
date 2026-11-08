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

## 本轮修好的一个真实 bug

**`embed` 表被硬编码搬进显存，而规划器按「它在主机」记账。**

```python
# engine.py 第 668 行（修复前）
self.embed_on_cpu = False      # 硬编码
```

而 `resources.plan` 一直在算 `embed_on_cpu=True`。于是规划器少算 1.23GB，
8GB 卡上多给了 5 层常驻。实测：

| | alloc | free |
| --- | --- | --- |
| 修复前 | 6.18GB | 0.33GB |
| **修复后** | **4.95GB** | **1.50GB** |

省下的正是那张 1.23GB 的 embed 表。现在 `embed_on_cpu` 跟随规划结果，
`TI_EMBED_GPU=1` 仍可强制放回显存（CUDA 图捕获需要）。

## 但 40 层在 8GB 上仍然装不下

修完之后 free=1.50GB，第一次 prefill 仍然：

```
dequant fast path unavailable (CUDA_ERROR_OUT_OF_MEMORY from cuMemSetAccess)
RuntimeError: !handles_.at(i) INTERNAL ASSERT FAILED
```

1.50GB 不够一次 prefill wave（chunk=8 需要 ~1.6GB 的激活 + 反量化缓存）。

**结论：8GB 卡的实际上限是 11 层左右**（`layers=11` 时放得下，40 层不行）。
这不是 bug，是 8GB 显存和 40 层 x 164MB 权重的算术。

## 顺带发现的另一个问题

这个 checkpoint 的 `head` 和 `embed` 是**两张独立的 1.23GB 表**（形状相同、
前 4KB 字节不同），不是 tied 关系。所以两者都要占显存，规划器的 non-tied
假设是对的 —— 错的是引擎硬编码把两张都放进了显存（应该只放 head）。
