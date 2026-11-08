# 短输入 NaN：找到一处真实的路径缺失（2026-11-08 13:49）

## 新发现：`fp8_linear` 对 3 维单行输入没有路径

```
fp8_linear: no path for M=1 (is_cuda=True dim=3 M=1 block=32 w=(4608, 5120)
             K%16=0 N%16=0 native=True k=False xdt=torch.bfloat16 wdt=float8_e4m3fn)
```

`x.dim() == 3` 且只有 1 行时：

- 第 320 行 `xf = x.reshape(-1, d)` → 1 行
- `if xf.size(0) > 1` 不成立
- 落到第 344 行 `if xf.size(0) == 1:` → **抛错**

而它其实只是「1 行」。已修：3 维时把 `x` 也拍成 2 维再往下走。

**但这个修复没有解决 NaN** —— 那个报错是我手动调用 `moe.shared.forward` 时触发的，
正常 forward 路径不经过它。修的是真实的路径缺失，不是 NaN 的成因。

## 目前关于短输入 NaN 的全部事实

```
11 -> 11   first=True   second=True
11 ->  1   first=True   second=True      # 探针里正常
 1 -> 11   first=False  second=True
 1 ->  1   first=False  second=False
11 ->  4   first=True   second=False
```

**同一段代码、同样输入，不同探针给出不同结果。** 唯一的解释是有未定义行为。

## 已排除（累计 15+ 项）

attention / hc_post / hc_mixes / sids / `_aq_buf` / `_asc_buf` / `_gpu_y_buf` /
异步时序 / 专家权重 / 分批 / `_launch_keep` / `_bf16_cache` / `streamed 层` /
`reset` 的具体条目（跳过任一段都无效）/ CUDA graph（`gpu_layers < max_layers` 时
根本不捕获）/ 短序列本身（干净引擎下 N=1..11 全部正常）。

## 判断

这是**未定义行为**（很可能是某个 GPU kernel 在特定形状下的越界或未初始化读），
表现形式受观测方式影响。**用 Python 探针已经无法进一步定位。**

正确的工具是 `compute-sanitizer --tool memcheck`：

```
compute-sanitizer --tool memcheck python probe_min.py
```

它会直接指出越界地址和越界的 kernel。本机可以试，但 8GB 显存 + sanitizer 的
额外开销可能 OOM。

## 本轮实质产出

1. 修了 `fp8_linear` 的 3 维单行路径缺失（真实缺陷，独立于 NaN）
2. 修了 `_stream_layers()` 的 `gpu_layers` 越界索引（真实缺陷，独立于 NaN）

两处都是**逻辑上确实错的**代码，只是不是 NaN 的成因。
