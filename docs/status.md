# 引擎现状（2026-11-08 11:23）

## 一句话

40 层全模型端到端跑通，**输出正确**（「1+1 等于几」答 `2`）。8GB 卡上 0.27 tok/s，
原因是 14 个槽位对 240 个专家需求 = 100% miss；这不是代码问题，是显存物理上限。

## 实测

```
PREFILL 63.5s tokens=15 slots=14 loads=3600 evict=3586
DECODE  0.27 tok/s loads=3840 evict=3826 hits=0 bykey=14
OUT     '2'                                    <- 正确
```

## 已完成且验证过的

| 项 | 证据 |
| --- | --- |
| 端到端正确性 | 上面那行输出 `2` |
| 槽位池覆盖式分配 | `tests/test_slot_pool.py`（5 条）+ `test_slot_snapshot.py`（5 条）|
| 三级专家存储（显存/内存/磁盘）| `tests/test_expert_tiers.py`（5 条）|
| stage 去 memcpy | stage 175ms → 3.8ms，整步 499 → 276ms |
| 拼批正确性 | `tests/test_batch_consistency.py`：attention d==0.0、MoE<1e-2 |
| positions 约定统一 | `tests/test_positions_normalize.py`（5 条）+ `test_positions_contract.py` |
| 调度器 park/resume | `tests/test_admission.py`（5 条）+ `test_sched_e2e.py`（4 条）|
| KV 换出字节上限 | `tests/test_kv_park.py`（6 条）|
| 并发约束三条 | `docs/concurrency_limits.md`，全部实测 |
| 空闲预热让路 | `tests/test_warm_yields.py` |
| pin 覆盖率门 | `tests/test_pin_coverage_gate.py`（3 条）|
| 可观测性 | `/health` 的 `expert_slots` |

## 关键数字（全部本机实测）

| 量 | 值 | 含义 |
| --- | --- | --- |
| MoE 单层（专家全驻留）| 0.95ms | 计算本身只占 38ms/40 层 |
| MoE 单层（真实 decode）| ~80ms | 差 79ms 全在等搬运 |
| 单个 18.8MB blob H2D | 2.65ms（7.11GB/s）| 8GB 卡的硬件下限 |
| 相邻激活路由重合 | 5/6 | 局部性强，槽位保留划算 |
| 随机激活路由重合 | 2/6 | 对照 |
| 8 条会话专家并集 | 35/层 | 40 层需 1400 槽位 |
| 覆盖 92% 路由所需 | 96 个/层 | pin 在预算内覆盖不到一半 |

## 还没做完的（诚实清单）

1. **多卡 / EP**：`ep.py` 协议已测，但 `moe.py` 从不读 `ep_world`/`ep_rank`，
   所以 `TI_EP=1` 是空操作。本机只有一张卡，EP 等价于不启用，**无法验证**。
   接入需要的三步写在 `docs/ep_status.md`。
2. **压缩层拼批**：写入已改逐行 scatter，读侧加了逐行掩码，但**没有跨步测试
   脚手架**（单次 forward 不产生压缩组）。未验证。
3. **1M 上下文端到端**：压缩 KV 的容量和换出算过（6.91GB/条），但没真跑过。
4. **KV 换出接进 forward**：`KVPool.park/resume` 在调度层工作了，但 attention
   侧的 state 保存/恢复没有端到端验证。
5. **L20 真机**：关机中。所有「L20 上会怎样」的推算都基于本机实测外推。

## 对 L20 的预期（推算，未验证）

| 项 | 8GB 本机 | L20 44GB |
| --- | --- | --- |
| 槽位 | 14 | 1359 |
| 每步搬专家 | 240（100% miss）| 0（全驻留）|
| 每步 | ~3.7s | ~80ms |
| 单路 | 0.27 tok/s | ~12 tok/s |
| 并发 | 1 | 5-7 |
