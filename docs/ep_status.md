# EP（专家并行）现状：协议就绪，路径未接（2026-11-08）

## 做到了什么

`ep.py` 里的协议是完整且正确的，`tests/test_ep_protocol.py` 七条锁住：

- `ExpertRoute`：按容量把 384 个专家切成**连续块**分给各 rank。连续是要紧的 ——
  专家的 blob 在盘上按 id 顺序排列，连续块让每个 rank 的读是顺序的；轮转分配
  会让每个 rank 跳过一半 blob。
- 每一条专家恰好属于一个 rank，`block` 首尾相接覆盖全部。
- `reduce_partials`：各 rank 算的是**不相交**的专家子集，所以层输出是它们的**和**。
  单元素直接返回（单卡不该多一次逐元素加法）。
- `load_balance`：按实际路由频次重新分配专家，压低最忙的那个 rank。

## 没做到什么：MoE.forward 里没有接

`engine.py` 会把 `ffn.ep_world` / `ffn.ep_rank` / `ffn._ep_route` 设到 MoE 上，
但 **`moe.py` 里没有任何代码读这三个字段**。也就是说：

```
TI_EP=1 目前完全没有任何效果 —— 既不会分发，也不会算错，就是没实现。
```

engine.py 里那段注释说的是「会算错、输出 WILL be wrong」，实际情况比那更早一步：
过滤根本没写，所以连错都谈不上。

## 接进去需要什么

正确的形状是这样（不是把 active 过滤掉就完事）：

```python
# MoE.forward，EP 开启时
partials = []
for rank, dev in enumerate(route.devices):
    mine = route.mine(set(indices), rank)      # 这个 rank 拥有的专家
    if not mine:
        continue
    # 在 dev 上准备这个 rank 的槽位表与 kernel 参数，只算 mine 对应的那些 expert
    partials.append(_run_fused_on(dev, mine, ...))
y = reduce_partials(partials, target_device=xf.device)
```

三个具体障碍：

1. **`ExpertStore.fetch_async` 没有 device 参数**：一个 rank 的专家总会落在层所在
   的那张卡上。要 EP 就得让每个 rank 的槽位池落在自己的卡上（`VramSlotCache` 已经
   支持 `set_devices`，本机多卡路径也改过一轮，这里能用）。
2. **`moe_fused_act/down` 用 `slot_ids[e] * blob_bytes` 寻址**：每个 rank 有自己的
   槽位表就有自己的编号空间，参数要按 rank 分别组装 —— 不能复用同一份 `_sids_buf`。
3. **归约要跨卡**：`reduce_partials` 走 host 中转（或 peer access），每层一次、
   40 层一次 token。这个代价只有在「专家读盘/读内存是瓶颈」时才划算 ——
   L20 上（1359 槽位装得下 240 个专家，稳态零搬运）EP 的收益要重新算，
   不能想当然。

## 为什么现在不做

本机只有一张卡。EP 在这台机器上 world=1，任何实现都等于不启用，**无法验证**。
写一段不能验证的多卡代码，风险高于收益 —— 前几轮的教训（window buffer 没清零、
`start_pos` 与 `positions` 不一致）都是「写了但测不出来」累积出来的。

等 L20 机器回来，按上面的三步接，并用 `tests/test_multi_device.py` 的脚手架
（`set_devices` + `slot_stream`）验证每个 rank 的槽位真的在正确卡上。
