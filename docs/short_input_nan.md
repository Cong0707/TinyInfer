# 短输入 NaN 的真实规律（2026-11-08 13:46）

## 决定性数据（同一引擎，连续调用，不 reset）

```
11 -> 11   first=True   second=True
11 ->  1   first=True   second=True
 1 -> 11   first=False  second=True
 1 ->  1   first=False  second=False
11 ->  4   first=True   second=False
 4 -> 11   first=False  second=True
```

规律：**`first` 是否为 True 只取决于输入长度**

- 11 token -> first 总是 True
- 1 / 4 token -> first 总是 False

`second` 则是 True / False 混合 —— 说明它还受前一次调用的影响。

## 这与之前的 probe_short 矛盾

`probe_short.py`（每次循环先 reset）：N=1..11 全部 finite。
这里（不 reset、连续跑）：1 和 4 都 NaN。

**差别是 `reset()`。** 但上一轮我又发现「reset 不能修复」，因为
`prefill -> reset -> short` 仍然 NaN。

两者合起来说明：

- 干净引擎 + 短输入 => 正常
- 跑过一次之后 + 短输入 => NaN
- 跑过一次 + reset + 短输入 => 仍然 NaN（reset 救不回）

## 现在的判断

**短输入（< 约 8 token）在引擎「用过一次」之后会 NaN。**

这解释了很多之前零散的观察：

- `test_three_turns_are_all_finite`（11 token 的长 prompt）通过
- `test_output_is_not_degenerate`（"你好"= 1 token，在它之后跑）失败
- `probe_short.py` 通过是因为每次都 reset 且引擎刚建

## 下一步

短输入走的是哪条路径？代码里有：

```python
if T <= step:
    return self._forward_chunk(ids, start_pos, positions=positions)
```

以及 graph 相关：

- `fa7f43b graph 不再绑定 start_pos（freqs 改成固定 buffer）`
- `feba951 prefill(pos=0 无图) 允许 eager 一次，decode 必须有图`
- `0bc8bd5 decode 只在 graph 上跑，miss 直接报错`

**1 token 的输入会走 decode 路径（要求 graph）**，而 graph 是为某个固定
形状捕获的。11 token 的 prefill 之后，graph 可能被 prefill 的形状覆盖，
于是 1 token 的 decode 用了错的图。

验证：`TI_NO_GRAPH=1`（如果有这个开关）或临时禁用 graph，看短输入是否恢复。
