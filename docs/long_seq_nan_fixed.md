# 长序列 NaN 的真正根因：压缩 attention 的 rope 表形状（2026-11-08 14:56）

## 实测（修复前）

```
N= 256  finite=True   absmax=14.91
N=1024  finite=False  absmax=nan      <- 1K token 就挂
```

**没有任何异常抛出**，引擎的 except 把它吞成 NaN。

## 真正的异常（手动展开后）

```
File "compressed_attention.py", line 583, in forward
  fr = self._latent_freqs(_grp0, _ng, b, latent.device)
File "compressed_attention.py", line 332, in _latent_freqs
  grp = grp0.unsqueeze(1) + torch.arange(ng, device=device).unsqueeze(0)
RuntimeError: The size of tensor a (1024) must match the size of tensor b (512)
```

## 原因

`grp0` 有两种形状：`[b]`（decode，s=1）和 `[b, s]`（prefill，s>1）。

```python
grp0 = (1, 1024)          # s=1024
grp0.unsqueeze(1)         # -> (1, 1, 1024)
arange(ng).unsqueeze(0)   # -> (1, 512)        ng = 512
(1,1,1024) + (1,512)      # 广播失败
```

s=1 时 `(1,1,1) + (1,512)` 能广播成功，所以**短序列一直是好的**，
这个 bug 只在 s > ng 时暴露。

## 修复

压缩组起点只由**每行最后一个位置**决定（前面的位置已经压进历史组），
所以先取最后一列：

```python
_g = grp0 if grp0.dim() == 1 else grp0[:, -1]
grp = _g.unsqueeze(1) + torch.arange(ng, device=device).unsqueeze(0)
```

## 验证

```
修复前:  N= 256 OK    N=1024 NaN
修复后:  N= 128  finite=True  absmax=16.14
         N= 256  finite=True  absmax=18.56
         N= 512  finite=True  absmax=20.67
         N=1024  finite=True  absmax=19.18
         N=2048  finite=True  absmax=19.84
```

**2K token 全部正常，此前 1K 就 NaN。**

## 为什么这条很重要

用户的目标是 **1M 上下文**。而这条修复之前，**1K token 就走不通**。

而且它是**静默失败**：异常被 except 吞掉，上层看到的是 NaN logits，
不是崩溃。如果没有专门测长序列，这个问题会一直潜伏到真卡上跑长请求才暴露。

## 回归测试

`tests/test_latent_freqs.py`（2 条）：

- `_latent_freqs` 必须同时接受 `[b]` 和 `[b, s]` 两种 grp0
- 两种形状算出的 rope 表必须**逐位相同**（最后一列语义）
