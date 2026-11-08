# NaN 追查：已排除的部分（2026-11-08 12:18）

## 已确认排除

| 怀疑点 | 测法 | 结果 |
| --- | --- | --- |
| 测量污染 | 一进程一引擎 | 已排除（纪律见 measurement_discipline.md）|
| 显存不足 | n=5 时 free=1298MB | 已排除 |
| 第 1 层 hc_post | 逐段打点 | 正常，1.273 -> 1.211 -> 1.047 |
| engram | 单独调用 | 正常，1.273 -> 1.211，delta 0.86 |
| 第 0/1/2 层单层 | 手调 block.forward | 全部 finite |
| `_forward_chunk` 跑 3 层 | 直接调用 | **finite=True** |

`_forward_chunk(ids, 0)` 在 `layers=[0,1,2]` 时是 finite 的。5 层时 NaN。

## 一个我之前搞错的比较

我在手动循环里加了 `engram` 调用，结果 layer 0 输出变成 428（不加时是 1.273）。
原因是**第 0 层也有 engram 属性**（`engram_layer_ids=[1,14]` 并不意味着只有这两层
有 engram 对象，只是这两层的 engram 是「生效」的）。所以手动复现时加 engram
得到的数值和 `_forward_chunk` 不可比 —— 那个 428 是我探针的错，不是引擎的。

教训：**手动复现引擎路径时，必须逐行对照，不能凭印象「大概就是这样」**。

## 真正的分界

```
layers=[0,1,2]     _forward_chunk finite=True
layers=[0,1,2,3]   finite=True      (前面测过)
layers=[0,1,2,3,4] finite=False
```

所以 NaN 是**第 4 层**引入的（`_dump_h` 也显示第 4 层的 out 是 nan，in 是 808）。

## 下一步

第 4 层是 `MLACompressedAttention`，`ratio=2`，`is_kv_source=False`（consumer）。
它的输入 808 已经很大（正常应该是个位数），所以要在**第 3 层结尾就检查**：
第 3 层的输出是 808，而第 0 层是 1.273。放大发生在第 2、3 层。

关键问题变成：**为什么第 2 层把 1.047 放大到 940**。
第 2 层是 kv_source（写压缩 KV）。它的 `after attn` 只有 8.938，但块输出是 940 ——
所以放大在 `hc_post` 或 MoE 部分。
