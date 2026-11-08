# NaN 的数值根源：重尾分布 + rms_norm(eps=1e-20) + bf16（2026-11-08 12:15）

## 逐层证据链（`_dump_h` 钩子，5 层模型）

```
layer 0: in 0.453  -> out 1.273     finite
layer 1: in 1.211  -> out 1.047     finite
layer 2: in 1.047  -> out 940.000   finite    <- kv_source，跳 900 倍
layer 3: in 940    -> out 808.000   finite
layer 4: in 808    -> out nan       NOT finite
```

NaN 首次出现是第 4 层的**输出**，它的输入（808）还是有限的。
但数值在第 2 层就已经失控（1.047 → 940）。

## 第 2 层内部（手调，逐段）

```
L2 input         absmax=  222.000
after hc_pre     absmax=  197.000
after attn_norm  absmax=    0.754      <- rms_norm 压回 0.75
after attn       absmax=    8.938
rt.compress_kv   absmax=    1.875
rt.index_k       absmax=    2.500
```

单看 `after attn_norm` 是 0.754，看起来正常。但**分布的尾部**：

```
L2 input   p50= 0.043  p99=  48.3  p9999= 178   absmax= 222
hc_pre out p50= 0.055  p99=  87.5  p9999= 174   absmax= 197
```

中位数 0.04，99.99 分位 178 —— **重尾四个数量级**。
rms_norm 用全向量范数归一，范数被尾部主导，于是中位数量级的分量被压到 ~1e-3。
`rms_norm_eps = 1e-20` 在这个尺度下等于没有保护，bf16 的尾数（8 位）留不住这些值。

## 为什么「逐层手调正常、engine.forward NaN」

手调时我只跑到第 2 层就停了（`layers=[0,1,2]`），第 3/4 层根本没执行。
而 `engine.forward` 跑满 5 层，误差在第 3、4 层累积放大成 inf/NaN。
**这不矛盾** —— 之前把「手调正常」当成「层本身没问题」是误判。

## 是不是我引入的

不是。本轮所有改动都是显存记账与调度（`embed_on_cpu` 跟随规划、槽位预算迭代、
pin 覆盖率门、预热让路），**没有触碰 `hc_pre` / `rms_norm` / attention 的数值路径**。

## 真正的疑点

`L2 input absmax=222` 本身就异常。一个 5 层模型的残差流不该有 222 的量级。
第 1 层（engram 层）的输出是 1.047，第 2 层输入却是 222 —— 这个跳变发生在
**第 2 层的 attention 内部**（`hc_pre` 之前就 222 了，说明是第 1 层的输出经
`hc_post` 之后变成 222 的）。

下一步应该看 `hc_post` 对第 1 层输出的处理，而不是继续追 attention。
