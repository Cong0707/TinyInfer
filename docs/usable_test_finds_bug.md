# 端到端可用性测试抓到了真问题（2026-11-08 13:38）

## 新写的测试

`tests/test_engine_usable.py`：

1. `test_three_turns_are_all_finite` —— 连续 3 轮 reset→prefill→argmax，**通过**
2. `test_output_is_not_degenerate` —— 短输入（1 token）的 logits 不能退化，**失败**

失败原因：`logits 全 NaN`（不是退化，是 NaN）。

## 差异定位

| 场景 | 结果 |
| --- | --- |
| 干净引擎 → 1 token | finite |
| 11 token → reset → 1 token | finite |
| 3 个长 prompt（各 reset）→ 1 token | **NaN** |

单独复现两个用例的顺序不行，需要**连续跑 3 次长 prompt**。

## 这说明什么

累积效应。3 次 prefill 之后，某个状态（槽位、共享 buffer、计数器）累积到出错。

## 已排除（累计十几轮）

attention、hc_post、sids、_aq_buf、_asc_buf、_gpu_y_buf、异步时序、专家权重、
分批、_launch_keep、_bf16_cache、短序列本身。

## 下一步

把 probe_abc 扩展成「3 次长 prefill 各带 reset，然后短输入」，逐次检查是第几次之后坏。
如果第 3 次之后坏而第 2 次正常，就说明有**固定容量**被填满 —— 8GB 卡 11 个槽位，
每次 prefill 都要 11x6=66 个专家请求，三轮下来槽位表/引用计数可能没清干净。

## 这个测试的价值

它把「引擎能不能用」变成了可执行的断言。之前的临时探针都是单项验证，
**没有一个测试检查「连续服务多个请求」**。这条测试现在就挂在那里失败，
直到问题修好为止 —— 这比临时探针诚实。
