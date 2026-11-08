# NaN 源头锁定在 MoE routed 部分（2026-11-08 12:22）

## 完整证据链

```
L0 out         absmax=1.273
L1 out         absmax=1.047
L2:
  MoE input    absmax=0.828
  shared out   absmax=1.508      <- 正常
  MoE out      absmax=868.000    <- routed 部分
```

`hc_mixes` / `hc_pre` / `rms_norm` / attention / shared expert **全部排除**。
放大发生在 `moe.forward()` 的 routed 专家路径里。

## 一个形状矛盾（下一个要查的）

```
输入:          h_n2 = [1, 11, 4, 5120]
  reshape 后:  11 行 x 5120

_gpu_y_buf:    [2, 5120]          <- 只有 2 行
_last_ids:     34 个专家 id
```

topk=6、11 个 token 应该路由 66 个（11x6）；实际 `_last_ids` 只有 34 个，
而输出 buffer 只有 2 行。

**这三者不一致**，说明 `moe.forward` 内部对输入行数的处理有问题。
结合此前测过的「多行 batching」改动（按行切、dim0 拼、还原 [b,s,hc,d]），
很可能 11 行被错误地压成了 2 行。

## 为什么这和 NaN 相关

如果专家的加权和是在错误的行上累加的，数值会失控：11 个 token 的专家输出
被加到了 2 个位置上，每个位置承受约 5 倍的累加。这解释了 0.828 -> 868 的放大。

## 下一个探针

1. 在 `moe.py` 的 `_forward_gpu_only` 入口打点：`xf.shape`、`indices.shape`、
   `n_exp`、`_nrows`，确认行数在哪个环节从 11 变成 2。
2. 对照 `_gpu_y_buf` 的分配：它是按 `xf.shape` 分配的，2 行说明 `xf` 进来时
   只有 2 行。

## 注意

`shared.forward` 独立调用会抛 `UnboundLocalError: _gu`，`gate.forward(x, None)`
会触发 `device-side assert` —— 两者都依赖 MoE 主流程的隐式状态。
做这类拆解必须用 monkey-patch 拦截，不能裸调。
