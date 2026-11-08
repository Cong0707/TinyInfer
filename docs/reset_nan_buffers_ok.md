# reset 后单行 decode NaN：buffer 尺寸全部正确（2026-11-08 13:27）

## 反直觉的实测

```
prefill(17行): sids=[-1,-1,1,7,0,12,-1,-1]  bad=55  out_finite=True    <- 55 个无效槽位，输出正常
decode(1行):   sids=[5,4,1,2,0,3]           bad=0   out_finite=False   <- 槽位全有效，输出 NaN
```

**槽位有效性不是原因。** 反倒是 prefill 有 55 个未就绪槽位却正常。

## buffer 尺寸也全部对得上

```
prefill: rows=17 n_exp=102 nk=160  aq=87040 asc=2720  buf_aq=522240 buf_asc=16320
decode:  rows=1  n_exp=6   nk=160  aq=5120  asc=160   buf_aq=522240 buf_asc=16320
```

`_aqx = buf[:n_exp*5120]`，decode 是 `buf[:30720]`，`view(1,6,5120)` 正好 30720；
`copy_` 源 `aq` 是 5120 个元素，`expand` 到 `(1,6,5120)` 也是 30720。**匹配。**

`_ascx = buf[:n_exp*nk]`，decode 是 `buf[:960]`，`view(1,6,160)` 正好 960；
源 `asc` 160 个，`expand` 到 960。**匹配。**

## 所以问题不在这些 buffer

已排除：

| 项 | 证据 |
| --- | --- |
| sids 有 -1 | prefill 有 55 个 -1 仍正常 |
| `_aq_buf` / `_asc_buf` 尺寸或残留 | 尺寸精确匹配，清零也不影响 |
| `_sids_buf` 覆盖 `_miss_scan` | `_miss_scan` 内部有 `event.synchronize()` |
| `_gpu_y_buf` 复用 | 改成每批独立分配无效（已回退）|
| attention | 所有调用 `out_finite=True`，索引全在界内 |
| `hc_post` / `hc_mixes` | 逐段检查全部 finite |

## 剩下的唯一嫌疑

`self._launch_keep = (sids, weights)` —— 注释里写着：

> fa/fd 是用 driver API 直接 launch 的，torch 的 caching allocator 不知道这些
> kernel 还在读这块显存 …… 异步执行中的 kernel 于是从「槽位号」里读到 2^31
> 之类的垃圾。

`_launch_keep` 只钉住**最后一次** launch 的 `(sids, weights)`。
decode 时 `rows=1`，`weights` 只有 6 个浮点数；如果 kernel 在 `_launch_keep`
被下次调用覆盖前还没跑完，读到的是别人的数据。

## 下一个探针

在 `_launch_keep` 赋值后再加一个 `torch.cuda.synchronize()`，看 NaN 是否消失。
如果消失，就是 launch-keep 的生存期问题，修法是把 `_launch_keep` 改成列表
（保留最近几次）或直接每层各自持有一份。
