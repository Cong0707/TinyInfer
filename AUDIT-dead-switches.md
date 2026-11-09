
# tinyinfer 死开关 / 备用路径审计清单

- **范围**：`src/tinyinfer/**`（92 个 .py，31,262 行）
- **方法**：全量正则扫描 + AST 静态分析（不可达语句、常量假分支、重复 def）+ 调用点计数 + 人工核对调用链
- **日期**：2026-11-09
- **约束**：本次审计**没有修改任何代码，没有加载模型，没有跑 benchmark**。所有结论来自源码与注释里已记录的实测数据。
- **判定标记**：`[确认]` = 已核对调用链/定义，确定；`[怀疑]` = 静态证据充分但未运行时验证（例如「某函数只在死函数里被调用」这类传递性死亡）。

---

## 0. 结论摘要

| 项 | 数量 |
|---|---|
| `TI_*` 环境变量（唯一名） | **104** |
| 其中在注释/文档里被提到、但**代码从不读取**（幽灵开关） | **11** |
| 其中只做调试打印、不改变计算路径 | 约 30 |
| `if (False):` 死代码块 | 1（32 行） |
| AST 判定为不可达的语句 | 1 |
| 定义了但**全包内零引用**的函数/方法 | 108 / 1268 |
| 未被任何模块 import 的孤儿模块 | 6 |
| 同一功能的多套实现（已列出） | 10 组 |
| `TODO` / `FIXME` | **0** |

最危险的不是死代码，是**「开关看起来在、实际没跑」**，以及**同一个开关在不同文件里有相反默认值**。这两类下面占了大半。

开工前先纠正任务说明里两处行号（已逐行核对，不是笔误推断）：

- 「`TI_AUTO_SLOTS` 默认关」在 **engine.py:1188**，不是 1131。
- 「注释说扩容由 `try_grow()` 完成」在 **loading.py:2234**，不是 engine.py:2234。engine.py 里提到 `try_grow` 的是 1176 / 1183 / 1204 三行。

---

## P0 — 默认关着的关键路径（代码在但没跑）

### P0-1 MTP 投机解码：三重断链，规划器还替它占着显存

- **位置**：engine.py:1040（加载开关）、engine.py:3913（唯一会用它做推理的函数）、engine.py:328（显存预留）、tiers.py:102 / tiers.py:118（tier 声明）
- **开关**：`TI_MTP`，**默认 "0"**（engine.py:1040）
- **现状**：
  - tier 2（AGGRESSIVE，Engine 的默认档，engine.py:151）与 tier 3 都写 `mtp=True`（tiers.py:102、tiers.py:118）。
  - engine.py:1026-1027 已经算出 `_mtp_ok = tier.mtp and hw_plan.speculation`。
  - **但 engine.py:1040 的条件是 `_mtp_ok and device != 'cpu' and TI_MTP == '1'`** —— tier 说开、env 说关，结果永远不加载。
  - 更荒诞的一层：engine.py:328 `_prof["mtp_resident"] = bool(tier.mtp)` 把这 1.33GB **预留进显存规划**（engine.py:325-327 的注释自己写着「实测 8GB 卡上算 11 层占 4.15GB、实际占 6.21GB，差的就是这 1.33GB」），而这份预留对应的权重从来没被加载。
  - 就算手动 `TI_MTP=1` 加载成功，`speculative_generate()`（engine.py:3913）在包内**零调用**（唯一外部引用是 kcheck/mtp.py:48，测试用），`_draft()`（engine.py:3866）只被它调用。**加载了也不会有人用。**
- **影响**：三件事同时成立 —— 白占 1.33GB 显存（挤掉 stream 层与专家槽位）、tier 2 不做投机解码、规划器把「MTP 会常驻」写进了唯一一份容量账本。tiers.py:14 把「MTP scheduling」写成 tier 2 的定义特征。
- **建议**：二选一。要么修 shape、删掉 `TI_MTP` 让 tier 决定、并把 `speculative_generate` 接进 server 的解码循环；要么把 `mtp` 字段、`mtp_resident` 预留、`speculative_generate` / `_draft` / `load_mtp` 全删，tier 2 的声明里也不要再提。**不能保留现状**：engine.py:1028-1029、1037-1039 已经自己承认「默认关掉，等 shape 修好再开」，而 shape 一直没修。

### P0-2 `TI_AUTO_SLOTS` 有两个相反默认值，`try_grow()` 因此从不执行

- **位置**：engine.py:1188（引擎侧）、server.py:2491（服务端侧）、loading.py:2760（`try_grow` 本体）
- **开关**：同名 `TI_AUTO_SLOTS`，**两处语义不同**：
  - engine.py:1188 `== "1"` → **默认关**；关掉时 `try_grow()` 不被调用。
  - server.py:2491 `!= "0"` → **默认开**；决定 HTTP 并发槽位数。
- **现状**：默认路径下专家显存槽位停在 planner 的保守值，`try_grow()` 一次都不会跑。
- **影响**（全部来自代码自带注释，不是我推测）：
  - loading.py:2763-2766：docstring 明说「这是「显存尽可能占满」的落地点，**而它以前只存在于注释里**……槽位永远停在 planner 的保守值（本机 24 个 / 451MB），显存里一多半是空的，而每步要搬两百多个专家」。
  - engine.py:1180-1181：planner 给的槽位数只有这张卡能放的三四分之一，「显存空着不用 = 每步多搬两百个专家」。
  - engine.py:1169 写 **「（默认开）」**，engine.py:1183 紧接着写 **「默认关」**，engine.py:1187 又写「用 TI_AUTO_SLOTS=1 显式打开」。同一段注释三个互相矛盾的结论，说明这里被改过至少三轮。
- **建议**：先改名拆成两个（`TI_SLOT_GROW` 与 `TI_SERVER_BATCH_SLOTS`），然后把 engine.py:1196-1201 那套 reserve（`WS_BASE + 8*WS_PER_TOKEN + prefill_chunk*WS_PER_TOKEN`）验证一次真实峰值；通过就**删开关无条件执行**，不通过就删掉 `try_grow` 与 loading.py:2235 的死块，承认「槽位数由 planner 定」是唯一路径。现在这样等于把 tier 2 的主要收益挂在一个人人不会设的环境变量上。

### P0-3 `TI_SERVE_LOOP=0` 这条「保留的回退路径」会自死锁

- **位置**：server.py:1512（判断）、server.py:1513（再次取锁）、server.py:747（外层已取锁）、server.py:984（建的是 `threading.Lock()`）
- **开关**：`TI_SERVE_LOOP`，默认 "1"（批处理路径）
- **现状**：`generate()`（server.py:747）先 `with self._gen_lock`，随后 `_generate` → `TI_SERVE_LOOP=0` 时在 server.py:1513 又 `with self._gen_lock`。server.py:984 建的是**不可重入**的 `threading.Lock()`，请求线程永久阻塞。
- **影响**：这条被注释描述为「kept reachable……falling back is a behaviour choice, not a crash」（server.py:1507-1509）的路径，实际是一个挂死。而它正是 P0-4 唯一能到达前缀复用的门 —— 这个 bug 把整个功能锁死了。`[确认]`（锁类型与嵌套次序都已核对）
- **建议**：删掉 server.py:1512-1515 和它背后的 `_generate_locked` 整条链；删之前先把它承载的独占能力（P0-4）迁到批处理路径。**不要「修一下锁」了事** —— 一条默认关闭、打开就死锁、还没有任何自动化覆盖的路径没有存在价值。

### P0-4 多轮前缀复用与 per-session KV 快照，只在默认关闭的分支里实现

- **位置**：server.py:1668-1691（前缀复用判断）、server.py:1745-1752（写回）、server.py:1395（读盘恢复）、server.py:1440（落盘）、server.py:1655（`_bind_slot`）、server.py:1461（`_trim_expert_cache`）
- **开关**：`TI_SERVER_PREFIX`，默认开（`!= "0"`）—— **但只在 `_generate_locked` 里**。
- **现状**：默认路径（server.py:1516 的 `_generate_batched`）完全没有 prefix 逻辑；`_slot_kv` / `_kv_path` / `_load_slot_disk` / `_stash_slot` 的调用点**全部落在 `_generate_locked` 内**（唯一调用点：1655 与 1745）。
- **影响**：注释自己给了数据 —— server.py:1670-1671：「OpenAI chat 请求带整段对话，不这么做每一轮都要重 prefill 整段历史（实测 9 → 25 tokens 的 prompt 把 10s 变成 26s）」。默认部署形态下这个 2.6x 回归一直在生效。附带 per-session KV 落盘一起失效，进程重启后没有任何 KV 续航。
- **建议**：把 prefix 复用移进 `_generate_batched`（`Sequence` 已带 prompt，`sched.ServingEngine` 那边加一个 prefix 命中判断即可），然后**删掉 `TI_SERVER_PREFIX` 和整条 `_generate_locked`**。这是本清单里「默认是旧路径」代价最大的单条。

### P0-5 tier 写进 MoE 的三个字段，MoE 一个都不读

- **位置**：engine.py:1505、engine.py:1506-1507、engine.py:1508
- **开关**：`tier.official_gemv` / `tier.async_expert_fetch` / `tier.batched_scatter`（tiers.py:56 / 51 / 52；tier 1-3 全部声明为 True）
- **现状**：
  - `ffn._official_gemv` 在 `moe.py` 中**零读取**（全包唯一出现处就是写入点 engine.py:1505）。
  - `ffn._async_fetch` 同样是纯写入；`moe.py` 里没有任何 `_async_fetch` 引用。
  - `ffn._batched_scatter` 同上。
- **影响**：tiers.py:9-10 把「async expert fetch, batched scatter」列为 tier 1 的 production-proven 移植项，tiers.py:105/121 对 tier 2/3 也这么写。三者都是**文档级优化**：读代码会以为在跑，实际是三个写了没人看的属性。`[确认]`
- **建议**：要么在 `moe.py` 里真正接上（`batched_scatter` 的落点应该是 moe 里逐专家 `index_add_` 那段，值得单独做），要么把三个字段与 engine.py:1505-1508 三行一起删，并改掉 tiers.py 的 notes。**不要保留「写进去假装开了」的中间态。**

### P0-6 `wo_low_rank`：算了，赋值了，从不读

- **位置**：engine.py:255（`self._wo_low_rank = float(_approx.get('wo_low_rank') or 0.0)`），来源 tiers.py:125（tier 3 写 `'wo_low_rank': 0.75`）
- **现状**：`_wo_low_rank` 在整包内**只有这一处赋值、零处读取**（全量 grep 核对）。
- **影响**：tiers.py:16-18 把「low-rank output projection」写成 tier 3 的核心近似手段；engine.py:240-241 的注释也说「Wire the two that change which weights are read (skip) and how the FFN output is projected」。skip 确实接了（engine.py:245-252 + 2800 的分支），**low-rank wo 没接**。
- **建议**：删掉 engine.py:255 与 tiers.py:125 的 key，tier 3 文档同步改为「low-rank wo 未实现」。若要做，落点在 `deepseek/attention.py` 的 `wo_b` 投影，而不是在 Engine 里存一个浮点数。

### P0-7 只有死函数读取的 tier 字段：`graph_catalog` / `prefix_checkpoint` / `resume_sampling`

- **位置**：
  - `graph_catalog`：tiers.py:49 定义，tier 1/2/3 在 tiers.py:88 / 104 / 120 设了 `(128,256,512,1024,2048)`，**唯一读者是 engine.py:3608**，而它所在的 `warm_graph_catalog()`（engine.py:3607）**零调用点**。
  - `prefix_checkpoint`：tiers.py:54，唯一读者 engine.py:3622 / 3632，分别在 `save_checkpoint`（engine.py:3619）与 `load_checkpoint`（engine.py:3631）里 —— 两者包内零调用（仅 `scripts/examples/multi_turn.py:50/55` 用）。
  - `resume_sampling`：tiers.py:55，唯一读者 engine.py:3641，在 `resample_from_hidden`（engine.py:3637）里 —— 零调用。
- **影响**：tiers.py:8-11 把「prepared capacity catalog, immutable prefix checkpoints, resume sampling」三项列进 tier 1 的已移植清单。三项都是**只有死函数读的字段**，即「配置上声明、运行时不存在」。
- **建议**：三个字段 + 四个死方法一起删；除非 `scripts/examples/multi_turn.py` 是必须留的产品入口 —— 那就要把它们接进 server，而不是留在示例脚本里。

### P0-8 `TI_MMA_GEMM`：默认走慢路径，而它自己的理由已经作废

- **位置**：gemm.py:276（开关）、gemm.py:319（无条件的 tensor-core 路径）
- **开关**：`TI_MMA_GEMM`，**默认关**（`== "1"`）
- **现状**：`M >= 64` 走 mma 只在 env=1 时发生；而 gemm.py:319 对 `M >= 128` **无条件**走同一个 `mma_gemm.fp8_gemm_mma`。
- **影响**：
  - 性能：gemm.py:270-274 自带数据 `M=64 mma 128.3us vs 64 行 2.6ms → 20x`、`M=128 → 33x`。默认关掉的是 20-33x 的 prefill GEMM 加速。
  - 逻辑：gemm.py:263-267 给这个开关的理由是「prefill 和 decode 必须同算术，否则差 1 个 bf16 ulp 会让输出 token 变」（`[989,43,4571,5958] → [223,19,17,20]`）。但 gemm.py:319 已经在 `M>=128` 时无条件走了 mma —— **那个理由对真正的 prefill 主体早已不成立**，只剩 `64 <= M < 128` 这个小窗口还「保守」。
- **建议**：这个开关现在的作用是把一个不一致的中间状态固定下来。要么删开关、把阈值统一到 64（一次改干净），要么删掉 276 行那 5 行、承认 64..127 也交给 319 行。**保留 env 只是把「两套算术并存」这件事藏起来。**

### P0-9 `TI_GRAPH` 默认值 "batch"，但注释说默认走 plain

- **位置**：server.py:768（`environ.get("TI_GRAPH", "batch")`）、server.py:766-767（注释）、server.py:772-778（分支）
- **现状**：注释写「引擎自己的 generate() 是同一条前向……实测稳定在 295ms/token。**默认走它**，需要时再用 TI_GRAPH=1 试图」；代码默认值是 `"batch"` → 进 `_generate_batched`，不是注释说的 plain。
- **影响**：注释与实际默认路径不符，任何人按注释读代码都会读错默认行为。而且 `TI_GRAPH=1` 那条图捕获路径（server.py:780-824）注释自认「实测 10 次请求里 3 次 500」。
- **建议**：默认值与注释对齐，**并删掉 `_gmode == "1"` 那条**：一个自带 30% 500 率的分支不该留在生产路径上，应整段删除（`_decoder_for` server.py:871 一并）。

### P0-10 `TI_WARMUP`：默认关，代价写在注释里

- **位置**：engine.py:1148
- **开关**：`TI_WARMUP`，**默认 "0"**
- **现状**：关掉后第一次真实请求承担 kernel 编译。
- **影响**：engine.py:1140-1147 自述「第一次用户请求否则要付 ~90s 和 +6.8GB 编译器内存，而这发生在 forward 内部」。这是**首请求体验**问题，不是稳态性能问题，但它正是「启动后第一次对话像卡死」的直接原因。
- **建议**：这是本清单里我最倾向「换实现而不是删开关」的一条 —— 注释给的理由（「一次 warm-up forward 留下的状态 reset() 收不干净，实测第一个真实请求会回答不相关的问题」）是**正确性 bug**，不是「warm-up 不好」。正确做法是修 `reset()` 的状态泄漏，然后**删掉开关无条件预热**。在修好之前，这个开关应当标注为「已知 bug 的绕行方案」，而不是一个可选优化。

### P0-11 `TI_FP8_GEMV`：注释承诺 1.8x，代码里是一个 `pass`

- **位置**：gemm.py:252（注释承诺）、gemm.py:240-257（实现是 `try: pass`）
- **现状**：整包**没有任何一处 `environ.get("TI_FP8_GEMV")`**（全量核对）。注释说 `TI_FP8_GEMV=1 enables it`，但没有任何代码读它。
- **影响**：gemm.py:243-245 写得非常清楚：「The name this used to import has never existed …… the whole M=1 fast path was dead: all 248 fp8 linears per decode token fell through to the block_M=32 GEMM at ~96 GB/s while this kernel reaches ~160」。然后 251-252 说改成 env 开关 —— **但开关没写**。所以现状是：M=1 快速路径**仍然是死的**，只是死得比原来体面（从 ImportError 变成 `pass`）。
- **建议**：「承诺了没实现」的教科书案例。要么把 kernel 的 misaligned address 修掉并接上（这是 decode 上 248 次调用的 1.8x），要么把 240-257 整段删掉，并把「decode fp8 GEMV 未接入」写进 docs。**不要留一个指向不存在开关的注释**。

### P0-12 Engine 与 Backend 的默认 tier 不一致

- **位置**：engine.py:151（`tier: OptLevel | TierConfig = OptLevel.AGGRESSIVE`）、server.py:946 / 1767 / 2237 / 2314（`tier: str = "conservative"`）、server.py:3576（CLI 默认 "aggressive"）、server.py:3578（help 文本写 "default: balanced"）
- **现状**：四个地方四个说法。直接 `Engine(...)` → tier 2（prefetch / 流式 / 槽位全开）；直接 `TinyModelBackend(...)` 或 `build_backend("tiny", ...)` → tier 0（bf16 反量化、无 fp8 GEMM、无预取、`kv_bits=16` 双倍 KV、`expert_cache_gb=4.0`）；CLI → tier 2；CLI help → 说 tier 1。
- **影响**：tier 0（tiers.py:66-71）就是「plain PyTorch 参考路径」，engine.py:227-231 那条 `TI_TORCH_ONLY` 诊断路径关掉的就是它。**用库 API 起服务会默认跑在参考路径上** —— 这是「代码看起来在、实际没跑」最贵的形态：不是某个开关，是整个优化层。`[确认]`
- **建议**：把 `TierConfig` 默认值收敛到一处（`tiers.plan()` 的 `level=OptLevel.BALANCED` 默认），所有 backend 构造器去掉 `tier="conservative"`，改成 `tier: str | None = None` + `plan(None)`。server.py:3578 的 help 文本必须改。

---

## P0-B — 承诺了但根本不存在（幽灵开关 / 只写不读的字段）

这一类比死代码危险，因为**读代码会以为有开关可用**。全部用「全包 grep + ±1 行窗口内是否有 `environ.get`」双重确认。

### P0-B-1 幽灵环境变量（11 个，只出现在注释 / docstring / 文档里）

| 变量 | 出现位置 | 注释承诺 | 现实 |
|---|---|---|---|
| `TI_FP8_GEMV` | gemm.py:252, 253 | 打开 M=1 fp8 GEMV 快路径（~160GB/s vs 96） | 无读取，实现是 `pass`（见 P0-11） |
| `TI_FP8_PQ` | gemm.py:288 | 「restores the old path」（M=2..8 的 PQ kernel） | 无读取；gemm.py:289 自认那个模块已删 |
| `TI_MOE_FP4` | tiers.py:114, 131 | 「It is opt-in (TI_MOE_FP4=1)」 | 无读取；tier 3 实际走的是 `approximations['fp4_experts']` 这个字符串 |
| `TI_HC_MIXES_KERNEL` | deepseek/hc.py:466, 467 | 「=1 re-enables it for benchmarking only」 | 无读取；hc.py:473 已改成无条件走 kernel |
| `TI_HC_NOFUSE` | kcheck/hc.py:66-78 | 用于切换 `native/torch` 与 `default` 两条路径做对拍 | **hc.py 从不读它** → kcheck/hc.py:68-73 两次迭代跑同一条路径，**这个测试永远通过且什么都没验证**（见 P0-B-3） |
| `TI_SINKHORN_DEBUG` | deepseek/hc.py:241 | 「这个 except 只在 TI_SINKHORN_DEBUG 下才可见」 | 无读取；那里实际是 `raise`（hc.py:242） |
| `TI_CLAIM_REAL` | budget.py:556 | 「opt-in (TI_CLAIM_REAL=1) until the caches allocate from the claimed pool」 | 无读取；engine.py:497 已经**硬编码** `claim(..., real=True)` —— 文档过期，真实行为与文档相反 |
| `TI_PP_DEBUG` | loading.py:3981 | 「re-reads the same blob …… and compares bytes」 | 无读取，对照实现不存在 |
| `TI_MARLIN_CACHE` | deepseek/gemv_fp4_marlin.py:202-204 | 三行 benchmark 表（1 / 64 / 512 的 prefill 时间） | 无读取；`set_cache_capacity` 由 engine.py:385 用槽位数直接调 |
| `TI_TOP_K` | engine.py:780 | 打印建议「TI_TOP_K=N can trade quality for IO here」 | 无读取。真正存在的是 `TI_TOPK`（engine.py:232）—— **少一个下划线，用户照提示设置会静默无效** |
| `TI_FP4_ROWS` | kcheck/more.py:64 | 注释里当成 env | 实为 CUDA 侧常量（gemv_fp4_w4a8.cu 的每组行数）。**不是幽灵，但注释误导**，同样要改 |

### P0-B-2 只写不读的实例字段

- engine.py:255 `self._wo_low_rank` —— 见 P0-6。
- engine.py:1505 `ffn._official_gemv`、engine.py:1506 `ffn._async_fetch`、engine.py:1508 `ffn._batched_scatter` —— 见 P0-5。
- engine.py:267 / 283 `self._kv_bits_effective` —— 两处赋值，**零处读取**。真正生效的是 `self._kv_quant`。`[确认]`
- loading.py:4528 `PrefetchConfig.prefetch_to_vram` —— 字段存在，`Prefetcher` 内部从不引用（而 `Prefetcher` 本身也是死的，见 P3-1）。

### P0-B-3 空转的「对拍」测试（比没有测试更糟）

- **kcheck/hc.py:65-79**：`TI_HC_NOFUSE` 循环两次，但 `deepseek/hc.py` 从不读这个变量 → 两轮都跑默认路径，`check_close` 必然通过。`[确认]`
- **kcheck/hc.py:32-45**：`TI_FUSE` 循环 `('1','fused'), ('0','two-step')`，但 `TI_FUSE` 只在 **import 期**被 `deepseek/model.py:21` 读一次进 `_FUSE_HC_PRE`，循环里改 `os.environ` 对已 import 的模块无效；而 `hc._FUSED_PRE_NORM`（hc.py:273 / 302-311）只按 kernel 可用性决定，**根本不看 TI_FUSE**。所以 "two-step" 那一轮测的还是 fused。`[确认]`
- **建议**：两个 case 要么改成直接调用参考实现（hc.py 里有 `_hc_pre_ref` / `_hc_post_ref`），要么删掉。保留会给人「两条路径都验证过」的错觉。

---

## P1 — 死代码块

### P1-1 loading.py:2235 `if (False):` —— 32 行自动扩容，注释和 `try_grow` 双双接管

- **位置**：loading.py:2235-2266（块体），前因注释 loading.py:2232-2234
- **内容**：按 `mem_get_info` 的 free 减 512MB reserve 自动扩槽位，重建 `_state` / `_last_used` / `_key` / `_refs` / `_event` 五个数组。
- **为什么是死的**：注释自己说「这里**不**按 free 自动扩容……真正的自动扩容在模型就位之后由 `try_grow()` 完成」。而 `try_grow`（loading.py:2760）**又默认不会跑**（P0-2）。所以这个 `if (False)` 和它注释里指认的替代品，**两条路都没通**。
- **建议**：连同 loading.py:2226-2234 的注释一起删。这一段是「设计意图 + 被禁用的实现 + 指向的替代实现也被禁用」的三层考古现场。`[确认]`（AST 判定为唯一一处常量假分支）

### P1-2 gemm.py:240-257 —— `try: pass` 形式的死分支

- **位置**：gemm.py:229（条件）、gemm.py:240-257（`try` 体内唯一语句是 `pass`）
- **内容**：整个 M=1 fp8 GEMV 快速路径只剩一个 `pass`，外面套 `try / except: pass`。
- **影响**：这是「语法上存在的分支」——覆盖率工具看不出来，但每次 decode 的 248 次 fp8 linear 都白走一遍条件判断。且它是 P0-11 的现场。
- **建议**：整段删除（含 gemm.py:258-282 的 TI_MMA_GEMM 块，见 P0-8）。`[确认]`

### P1-3 gemm.py:416 —— `raise` 之后的 `return out`，`out` 未定义

- **位置**：gemm.py:412-416
- **内容**：`raise RuntimeError(...)`（412-415）之后是 `return out`（416）。AST 判定不可达；`out` 在该作用域没有定义（306 行的同名变量在另一个分支内）。
- **建议**：删掉 416 行。这是全包唯一被 AST 判定的不可达语句。`[确认]`

### P1-4 engine.py:1269 的 `_layer_major_mode` 分支永远不会成立

- **位置**：engine.py:1269（读）、engine.py:2952（写 True）、engine.py:2957（写 False）
- **内容**：唯一把它置 True 的地方在 `prefill_layer_major()`（engine.py:2936）内，**而 `prefill_layer_major` 全包零调用**。
- **建议**：随 `prefill_layer_major` / `_prefill_layer_major_impl` 一起删（见 P2-1）。`[确认]`

### P1-5 deepseek/hc.py:294 docstring 与实现不符

- **位置**：deepseek/hc.py:288-295（docstring）、hc.py:302-311（实际判定）
- **内容**：docstring 说「Set TI_FUSE=0 to force the two-step path」，但 `hc_pre_rms_norm` 的开关是模块级 `_FUSED_PRE_NORM`，只按 kernel 能否加载决定，**不读 TI_FUSE**。`TI_FUSE` 只在 `deepseek/model.py:21` 的 import 期被读进 `_FUSE_HC_PRE`（用于 model.py:368 / 392 选择「调这个函数」还是「自己拼两步」）。
- **影响**：直接调用 `hc.hc_pre_rms_norm` 的人会以为 `TI_FUSE=0` 有效，实际无效。这直接导致 P0-B-3 的空转测试。
- **建议**：docstring 改成指向 `model.py` 层的开关，或者在 `hc_pre_rms_norm` 的判定里真的把 `TI_FUSE` 纳进来（二者选一，但要与 kcheck 对齐）。

### P1-6 关于「被注释掉的实现」：本类别**基本为空**

按要求扫了 ≥15 行的连续注释块，命中 19 处：engine.py:395-412 / 415-464 / 528-556 / 570-587 / 607-643 / 684-714 / 1161-1187 / 1879-1912 / 2199-2216 / 2816-2836 / 3058-3072 / 3457-3471、loading.py:2876-2897、resources.py:587-603、gemm.py:258-274、deepseek/attention.py:364-383、deepseek/hc.py:458-472、tp.py:165-179、chat.py:1-16。

**逐块核对结论：没有一块是「注释掉的旧实现」，全部是设计推理与实测数据记录。** 这个项目的问题不是「注释掉代码」，而是**注释越写越长、结论互相矛盾**（典型：engine.py:1169 vs 1183 vs 1187 三句话三种默认值；engine.py:455「池子的大小由规划器给，这里不再自己算第二套」vs 475 行仍在算）。

`[确认]` 这条结论对清理策略很重要：**不需要做「取消注释」式的恢复工作**，需要做的是把注释里已经过期的历史决策删掉，让每处注释只描述当前唯一路径。

---

## P2 — 从未被调用的函数 / 方法

判定口径：**全包（92 个 .py）内零引用**，且已排除 dunder、`@property` / `@x.setter` 对，以及由装饰器注册的 FastAPI 处理器。共 **108 个**（1268 个 def 中）。下面按「直接删」和「需要判断」分组。

### P2-1 engine.py（12 个）

| 位置 | 名字 | 备注 |
|---|---|---|
| engine.py:1432 | `expert_device` | 专家设备查询，零调用 |
| engine.py:1573 | `staticize_decode` | **内部还有 `self.enable_prefetch = False`（engine.py:1605）与 `self.predictor = None`（1606）—— 关掉预取的「正式入口」本身是死的** |
| engine.py:1965 | `save_kv_cache` | 与 engine.py:1978 `load_kv_cache` 成对，都零调用；server 用的是自己那套（server.py:1395 / 1440） |
| engine.py:1978 | `load_kv_cache` | 同上 |
| engine.py:2525 | `_ensure_slots` | 唯一调用者 `_slot_for`（2530）本身零调用 → 整链死 |
| engine.py:2530 | `_slot_for` | 「两层权重搬进轮转显存槽」机制，docstring 写得完整，实际从未执行 |
| engine.py:2553 | `_stream_into_slot` | 配合 `_slot_for` |
| engine.py:2936 | `prefill_layer_major` | 层主序 prefill 入口，零调用；实现 `_prefill_layer_major_impl`（2960）只被它调用 |
| engine.py:3607 | `warm_graph_catalog` | 唯一读 `tier.graph_catalog` 的地方（P0-7） |
| engine.py:3619 | `save_checkpoint` | 唯一读 `tier.prefix_checkpoint`；仅 `scripts/examples/multi_turn.py:50` 用 |
| engine.py:3631 | `load_checkpoint` | 同上（脚本 55 行） |
| engine.py:3637 | `resample_from_hidden` | 唯一读 `tier.resume_sampling` |

补充：engine.py:3913 `speculative_generate` 包内零调用（只有 kcheck/mtp.py:48 用），engine.py:3866 `_draft` 只被它调用 —— 与 P0-1 同一件事。

### P2-2 loading.py（27 个，重灾区）

| 位置 | 名字 | 备注 |
|---|---|---|
| loading.py:104 | `layer_expert_bytes` | 容量查询族 |
| loading.py:110 | `token_expert_bytes_with_engram` | 同上 |
| loading.py:327 | `slots_for_phase` | 只在 `docs/设计-专家加载.md:419` 被提到 |
| loading.py:393 | `is_resident` | 只在 docs:420 被提到 |
| loading.py:573 | `pseudo_code` | 生成伪代码字符串，只被 docs:422 引用 —— **文档工具函数进了运行时模块** |
| loading.py:940 | `layer_of_shard` | — |
| loading.py:1147 | `read_many_pinned` | 与活着的 `_read_many_pooled` / `_read_misses_iocp` 重复 |
| loading.py:1668 | `pinned_entries` | 统计用 |
| loading.py:2079 | `unpin` | 只在 docs:416 被提到 |
| loading.py:2673 | `plan_quota` | 配额规划，零调用 |
| loading.py:2694 | `reset_counts` | — |
| loading.py:2962 | `_find_victim_locked` | 淘汰策略内部函数，零调用（在用的是 `_claim_slot_locked` 的游标版） |
| loading.py:3075 | `table_set` | 与活着的 `table_sync` / `table_flush` 重复 |
| loading.py:3469 | `request_queue` | — |
| loading.py:3477 | `_gpu_addressable` | — |
| loading.py:3850 | `_pin_cap` | **内部读 `TI_PIN_CACHE_GB`（3859）—— 一个死函数里的活开关** |
| loading.py:3870 | `hot_prefetch` | 「把最热专家读进 RAM」，零调用；注释说 called between requests，但没有请求间钩子 |
| loading.py:4086 | `start_warm` | **与 `expert_tiers.py:233` 的同名实现重复，活的是后者**（engine.py:765 调的是 `_mgr.start_warm()`，`_mgr` 是 `ExpertTierManager`） |
| loading.py:4134 | `stop_warm` | 同上重复；engine.py:74-75 调的是 `ExpertTierManager.stop_warm` |
| loading.py:4139 | `fetch_async` | 异步读盘入口，零调用（engine.py:1324 只是注释里提到它） |
| loading.py:4170 | `drop_slots` | **prefill 前释放槽位**的保护，零调用 → `_paused` 永远不会被置 True，整个 `_paused` 机制退化成恒 False |
| loading.py:4219 | `resume_slots` | 与 4170 成对 |
| loading.py:4438 | `load_npz` | 路由表持久化 |
| loading.py:4447 | `save_npz` | 同上 |
| loading.py:4479 | `set_routes` | 喂给预测器的路由，零调用 |
| loading.py:4607 | `note_step_start` | 预测器步进钩子，零调用 |
| loading.py:4054 | `prefetch_to_vram` | 只在同文件 `_selftest_body`（loading.py:5087）里用 |

### P2-3 其余模块（节选，全部零引用）

- budget.py:182 `cpu_address`、budget.py:215 `_pinned_pool`
- caps.py:74 `warp_specialized`、caps.py:116 `all_caps`（多卡取最差能力，多卡路径没用上）
- cpu_kernels.py:26 `staging_buffer`、cpu_kernels.py:62 `stage_tensor_to_device`
- deepseek/hc.py:186 `_sinkhorn_graph`、deepseek/hc.py:200 `_sinkhorn_fn`
- deepseek/moe.py:111 `_swiglu2`、deepseek/moe.py:170 `forward_fp4`、deepseek/moe.py:204 `forward_fp4_pq`（**tier 3 声明 `fp4_experts: True`，但这两个 fp4 专家路径零调用**）、deepseek/moe.py:416 `_req_epoch`、deepseek/moe.py:456 `_load_experts`、deepseek/moe.py:898 `_to_acc`
- gemm.py:50 `_bf16_default`、gemm.py:88 `fp8_linear_q`
- kernels/cutlass_gemm.py:46 `fp8_blockwise_gemm`（整个 cutlass 分支死）、kernels/native.py:155 `stream_write_value32`
- kv_tiers.py:39 `groups_that_fit`、kv_tiers.py:48 `sessions_that_fit`、kv_tiers.py:93 `plan_expert_slots`（仅 tests 用）—— **kv_tiers 里只有 `plan_slots` 与 `batch_slots_bound` 是活的**
- placement.py:23 `prefer_cpu`（在用的是同文件 `small_op_on_cpu`，engine.py:830 只 import 后者）
- quant.py:27 `fp8_block_scale_to_dense`
- qwen/qwen_kernels.py:97 / 132 / 179 `attention_layer` / `linear_layer` / `model_forward`；qwen/qwen_model.py:357 / 365 / 384 / 392 / 496 `_touch` / `_evict_one` / `forget_layer` / `_host_kv` / `capture_step`（Qwen 的显存淘汰与图捕获都没接上）
- server.py:2036 `session_count`；server.py:2137 / 2181 / 2195 `_check_stop` / `_content_parts` / `_check_messages` —— **后三个是 pydantic validator，由 `@field_validator` 注册，属误报，删前必须确认** `[确认：误报]`
- tp.py:430 `build_sharded`、weights.py:97 `shard_files`、weights.py:202 `get_slice_info`
- expert_tiers.py:306 `warm_progress`（预热进度从未报告给用户）
- protocol.py:78 `check_model`（整个模块无人 import，见 P2-4）

### P2-4 孤儿模块（全仓库零 import，已排除 `.venv`）

| 模块 | 大小 | 判定 |
|---|---|---|
| `kv.py` | 5,365 B | **零引用**。KV 实际实现在 `deepseek/compressed_attention.py` + `kv_tiers.py`（规划）里 |
| `scheduler.py` | 6,765 B | **零引用**。EEVDF 调度器；真在用的是 `sched.py`（server.py:1535 / 1587） |
| `mlp_fused.py` | 2,411 B | **零引用** |
| `protocol.py` | 3,592 B | **零引用** |
| `approx_math.py` | 2,624 B | 仅 `tests/test_approx_math.py` 用；引擎的 `fast_math` 实际只设了两个 TF32 全局开关（engine.py:261-265） |
| `draft.py` | 2,288 B | 仅 `tests/test_draft.py` 用；引擎自己有一套 `_draft`（engine.py:3866，也是死的） |

`[确认]` 六个模块合计约 23KB、约 600 行，是「落了地但没入户」的独立实现。它们让「这个功能到底在哪」必须从一次考古开始。

---

## P3 — 同一功能的多个实现

每组都标了**当前**（实际执行）与**遗留**（不执行）。

### P3-1 专家预取：4 套

| 实现 | 位置 | 状态 |
|---|---|---|
| 逐层滚动（RAM 预热） | engine.py:2561 `_prefetch_layer` ← engine.py:2718 | **当前**（受 `enable_prefetch` 控制） |
| 逐层滚动（搬进显存槽） | engine.py:2575 `_warm_next_layer` ← engine.py:2882 | **当前**（同一开关，但另有 5 个 early-return：2575-2606 的 `slots <= 0` 等 → 8GB 卡上恒返回 0） |
| `ExpertAsync` / `ExpertQueue` 流水线 | deepseek/moe.py:379 `_async_experts` ← moe.py:664 | **当前**，但走的是**另一个开关** `TI_ASYNC_EXPERTS`（moe.py:388，默认 "1" = 开） |
| `Prefetcher` + `OraclePredictor` + `EnsemblePredictor` + `PrefetchConfig` | loading.py:4540 / 4517（及其上方的 predictor 类） | **遗留**，只在 `loading._selftest_body`（5059）里被实例化 |

另外**前瞻深度**也有两套同时生效、互不一致：

- engine.py:2715 `_depth = tier.prefetch_k`（tier 2 = 2、tier 3 = 3）控 RAM 预热层数；
- engine.py:2752-2753 `_adv = self._prefetch_depth`（PID 控制器 `autotune.PrefetchController`，engine.py:909）控槽位预取层数。

同一个「提前几层」的概念被拆成两个旋钮，只有一个会随运行时反馈变化。**建议**：`_prefetch_layer` 与 `_warm_next_layer` 合成一个「把后续 N 层的专家搬到该去的地方」的函数，深度只保留 PID 一个来源；`loading.Prefetcher` 全家（含 `OraclePredictor` / `EnsemblePredictor` / `PrefetchConfig`，约 15KB）整段删除。

### P3-2 显存槽位：5 套（含 3 种「还剩多少显存」算法）

| 实现 | 位置 | 状态 |
|---|---|---|
| `VramSlotCache`（专家槽 + arena） | loading.py:2139 起 | **当前** |
| `SlotManager`（会话 KV 槽） | server.py:1910 | **当前**（另一层含义的槽位） |
| engine「整层权重搬进轮转设备槽」 | engine.py:2525 / 2530 / 2553 | **遗留**（零调用，P2-1） |
| `_ensure` 里的自动扩容 | loading.py:2235 `if (False)` | **遗留**（P1-1） |
| `try_grow` | loading.py:2760 | **存在但默认不跑**（P0-2） |

「还剩多少显存」的推算散在四处：loading.py:2788（`mem_get_info`）、server.py:2500（又一次）、server.py:1607（第三次）、`resources.plan()`（规划器）。**建议**：显存余量只允许一个来源（规划器），另外三处改成读规划器结果或删掉。

### P3-3 空闲预热线程：2 套

- **当前**：`expert_tiers.py:233 ExpertTierManager.start_warm`（engine.py:765 通过 `_mgr` 调用；停止走 engine.py:74-75）。
- **遗留**：`loading.py:4086 ExpertStore.start_warm` + `loading.py:4134 stop_warm` + `loading.py:4093 _warm_thread`，零调用。

两套注释几乎逐字相同（「内存放得下整个专家集……预热完之后每次 decode 只剩 RAM->VRAM 的 DMA」），是同一段设计被复制到两个类里。

### P3-4 调度器：2 套

- **当前**：`sched.py`（`ServingEngine`，server.py:1535 / 1587）+ `sequence.py`。
- **遗留**：`scheduler.py`（EEVDF：`charge` / `next_quantum` / `prefill_caps` / `decode_caps`），**全仓库零 import**。

### P3-5 KV 缓存：4 套概念

- `deepseek/compressed_attention.py` 的 `MLACompressedAttention` —— 真正在跑。
- `kv_tiers.py` —— 规划：只有 `plan_slots` / `batch_slots_bound` 被 server 用；`groups_that_fit` / `sessions_that_fit` / `plan_expert_slots` 死。
- `kv.py` —— 独立实现，**零引用**。
- server 里的 per-session KV 落盘（`_kv_path` / `_load_slot_disk` / `_slot_kv`，见 P0-4）—— 挂在默认关闭的路径上。

### P3-6 投机解码：3 套

- `draft.py`（零引用，仅测试）。
- `engine.py:3866 _draft` + `engine.py:3913 speculative_generate`（包内零调用链，仅 kcheck 用）。
- `deepseek/mtp.py`（`load_mtp` / `forward_with_main` / `main_hiddens` 等，默认不加载，见 P0-1）。

三套里没有一套在默认路径上跑。

### P3-7 快速数学：2 套，且配置项只有布尔语义

- **当前**：engine.py:256-265 —— 只是打开 `torch.backends.cuda.matmul.allow_tf32` 与 `cudnn.allow_tf32`，注释自己承认「there is no torch-level switch for one op at a time」。
- **遗留**：`approx_math.py`（`fast_rsqrt` / `fast_silu` / `poly_sin` / `poly_cos`），零运行时引用，只有 tests。
- **附**：tiers.py:127 的 `fast_math: ['exp','silu','rsqrt','sin','cos']` 被读进 `self._fast_math`（engine.py:256），**然后只用来判断「非空就开 TF32」** —— 具体是哪五个 op 完全没区别。这是一个「配置项看起来很细、实际只有一个布尔语义」的例子。

### P3-8 权重上卡：4 条路里只有 1 条常年跑

- **flat ring**（当前，`_flat_setup` / `_flat_apply` / `_flat_wait` / `_flat_prefetch`，engine.py:2103-2478，`TI_FLAT`）
- **v5 parity stream**（`_ensure_stream_buffers` / `_stream_copy_sync` / `_stream_prefetch`，engine.py:1988-2524，`TI_V5=1` 才走）
- **per-tensor `.to(device)`**（engine.py:2778 的 `b.to(self.device)` 兜底）
- **`_slot_for` 的整层轮转槽**（engine.py:2530）—— 死代码

### P3-9 同名方法在同一作用域定义两次（被静默覆盖）

| 位置 | 方法 | 说明 |
|---|---|---|
| `expert_tiers.py:59` 与 `expert_tiers.py:156` | `ExpertTierManager.table` | 两个实现语义等价（都返回 `slots.slot_table(...)`），**156 行覆盖 59 行** |
| `kernels/native.py:374 / 386 / 411` 与 `kernels/native.py:446 / 458 / 463` | `set_dynamic_smem` / `bind` / `launch` | **整段约 70 行被复制粘贴两次**；常量 `_CU_FUNC_MAX_DYNAMIC_SMEM` 也重复定义（371 与 443）。后一份覆盖前一份，前一份是纯死代码 |

`[确认]`（AST 扫描 + 人工比对）。deepseek/attention.py 与 deepseek/compressed_attention.py 里的同名重复是 `@property` / `@x.setter` 对，**不是问题**，已排除。

### P3-10 缓存淘汰策略：3 套实现、2 个默认值

- 实现：`LRUPolicy`（loading.py:1264）、`ARCPolicy`（1293）、`S3FIFOPolicy`（1383）。
- `ExpertCache.__init__` 默认 `policy="arc"`（loading.py:1622、3662），**但 engine.py:681 传的是 `environ.get("TI_CACHE_POLICY", "lru")`** → 生产默认其实是 **lru**。
- 结论：**ARC 与 S3FIFO 在生产路径上永远是死代码**（只有 `loading._selftest_body` 与 `scripts` 里的模拟用过）。engine.py:679-680 已说明为什么不用（「this model's routing is flat enough that they mostly cost capacity」）。
- **建议**：删掉 ARC / S3FIFO 两个类与 `TI_CACHE_POLICY`，`ExpertCache` 默认值改成与引擎一致（lru）。现在这样，读 `loading.py` 的人会以为默认是 ARC。

---

## 附录 A — `TI_*` 全表（104 个）

「默认」一栏取代码里的字面默认值；「建议」一栏只对**行为开关**给，调试类统一在 A-2 结尾给结论。

> 分类之间**故意重叠**：一个变量可能既改变执行路径、又只用于打印，幽灵变量也常同时出现在 A-4。所以四张表**不必相加等于 104** —— 它们回答的是「这个变量属于什么性质」，不是「数量对账」。

### A-1 改变执行路径的开关

| 变量 | 读取位置 | 默认 | 开 / 关 分别走什么 | 建议 |
|---|---|---|---|---|
| `TI_AUTO_SLOTS` | engine.py:1188 | 关 | 开：`try_grow()` 扩槽；关：planner 静态值 | **删**（P0-2） |
| `TI_AUTO_SLOTS` | server.py:2491 | 开 | 开：按 free 算批宽；关：用 `max_slots` | 改名（同上） |
| `TI_MTP` | engine.py:1030、1040 | 关 | 开：加载 DSpark 三阶段；关：不加载 | **删**（P0-1） |
| `TI_ASYNC_FETCH` | engine.py:1507 | 关 | 开：`ffn._async_fetch=True`（**但没有代码读它**） | **删**（P0-5） |
| `TI_ASYNC_EXPERTS` | deepseek/moe.py:388 | 开（"1"） | 开：走 `ExpertAsync` 流水线；关：同步 | 保留（但应与 P0-5 合并成一个概念） |
| `TI_PREFETCH` | engine.py:898 | 未设 → 取 tier | 开：预取；关：不预取 | 删（`tier.prefetch` 已能表达） |
| `TI_WARM` | engine.py:752 | 开（"1"） | 开：后台预热专家进 RAM；关：不预热 | 保留 |
| `TI_WARMUP` | engine.py:1148 | 关（"0"） | 开：启动期跑一次 forward 编译 kernel；关：首请求付 JIT | 修 `reset()` 后删（P0-10） |
| `TI_SERVE_LOOP` | server.py:1512 | 开（"1"） | 开：`_generate_batched`；关：`_generate_locked`（**自死锁**） | **删**（P0-3） |
| `TI_SERVER_PREFIX` | server.py:1672、1746 | 开（!="0"） | 开：多轮前缀复用；关：整段重 prefill | 迁进批处理路径后删（P0-4） |
| `TI_GRAPH` | server.py:768 | "batch" | "plain"→`_generate_plain`；"1"→图捕获；其它→批处理 | 删 "1" 分支（P0-9） |
| `TI_MMA_GEMM` | gemm.py:276 | 关 | 开：64≤M 走 tensor-core；关：逐行 GEMV | 删（P0-8） |
| `TI_PREFILL_TL` | gemm.py:301 | 关（!="1"） | 未开：prefill 逐行走与 decode 同 kernel；开：跳过行循环落到 M≥128 分支 | 删（注释指向的 tilelang 已删除） |
| `TI_FP4_MARLIN` | gemm.py:383 | 关（'0'） | 开：M≤8 先试 Marlin w4a16；关：native w4a8 | 删（gemm.py:369-381 已说明 Marlin 会改答案） |
| `TI_MULTIROW_GEMV` | deepseek/moe.py:229 | 关 | 开：多行走 `gemv_fp4_m8` | 删 |
| `TI_HC_GPU` | engine.py:835 | 关（!="1"） | 关：小算子放 CPU；开：留 GPU | 删（应自动判断，见附录 B） |
| `TI_EMBED_GPU` | engine.py:965 | 关 | 开：embedding 放显存（旧行为）；关：按规划 | 删（规划器已有结论） |
| `TI_HEAD_CPU` | resources.py:529 | 关 | 开：强制 head 在 CPU | 删（见附录 B-6：打开必坏） |
| `TI_HEAD_KERNEL` | engine.py:2620 | 开（!="0"） | 关：不走 head kernel | 删（改成 try/except） |
| `TI_RESIDENT` | resources.py:538 | 未设 | 设值：强制常驻层数 | 删（破坏槽位规划，见 resources.py:617-619） |
| `TI_MAX_RESIDENT` | resources.py:573 | 未设 | 设值：常驻层上限 | 删 |
| `TI_GPU_LAYERS` | engine.py:821 | 规划值 | 覆盖常驻层数 | 删（engine.py:819 自称 overrides for experiments） |
| `TI_NO_SLOTS` | engine.py:786 | 关 | 开：关槽位池换确定性 logits（注释称代价约 40% decode） | 改名并修 bug（附录 B-3） |
| `TI_ALL_PINNED` | engine.py:1717 | 关 | 开：强制走全命中路径（量上限） | 删（纯测量） |
| `TI_SLOTS` | engine.py:375 | 规划值 | 覆盖专家槽位数 | 保留（压测固定） |
| `TI_RING` | engine.py:2179、resources.py:448 | "3" | 环形缓冲深度 | 改由 `_shed_vram` 单向收缩（附录 B-5） |
| `TI_FLAT` | engine.py:567、2046 | 未设（自动） | "0"：不用 flat slab；未设：按是否全常驻自动决定 | 保留 |
| `TI_CHUNK` | engine.py:827 | 规划值 | 覆盖 prefill chunk | 删（与 `TI_PREFILL_CHUNK` 重复） |
| `TI_PREFILL_CHUNK` | engine.py:799 | 未设 | "1"：逐 token prefill；未设：用规划 chunk | 保留（正确性对拍） |
| `TI_TOPK` | engine.py:232 | "0" | >0：覆盖 `tier.expert_topk` | 保留（质量/IO 换取） |
| `TI_TIER` | engine.py:36 | 未设 → AGGRESSIVE | 0-3 选档 | 保留（但先修 P0-12） |
| `TI_EP` | engine.py:1330 | 关（!="1"） | 开：专家并行（**注释自认答案会错**） | 删（engine.py:1343-1345 已警告 Output WILL be wrong） |
| `TI_DIRECT` | engine.py:715 | 关（"0"） | 开：NO_BUFFERING 直读；关：走页缓存 | 保留（附录 B-1） |
| `TI_ZC` | engine.py:2147 | 关 | 开：zero-copy slab 布局实验 | 删（engine.py:2143-2145 自认地址算术不可信） |
| `TI_V5` | engine.py:2659、2775、2962 | "0" | 开：parity stream prefill | 删（入口 `prefill_layer_major` 已死） |
| `TI_NO_FLAT_PREFETCH` | engine.py:2740 | 关（!="1"） | 开：停 flat ring 预取 | 删（PID 已能调深度） |
| `TI_CACHE_POLICY` | engine.py:681 | "lru" | lru / arc / s3fifo | 删（P3-10） |
| `TI_CACHE_GB` | engine.py:596 | 未设 | 硬覆盖宿主专家缓存字节 | 与 `TI_CACHE_CAP_GB` 合并成一个 |
| `TI_CACHE_CAP_GB` | engine.py:644 | 未设 | 硬覆盖缓存上限 | 同上 |
| `TI_SLAB_FRAC` | engine.py:2217 | "0.28" | 层 slab 占物理内存比例上限 | 移入规划器（附录 B-2） |
| `TI_HOST_POOL_GB` | engine.py:476 | 未设 | 硬覆盖 pin 池大小 | 保留 |
| `TI_RAM_FRACTION` | budget.py:524 | 未设 | 覆盖 `HostBudget.limit` | 保留 |
| `TI_RAM_RESERVE_GB` | budget.py:895 | 未设 | 覆盖 `HostBudget` 的 reserve | 保留 |
| `TI_CLAIM_TOUCH` | budget.py:751 | 关（"0"） | 开：claim 后立即 touch 锁页 | 合并成 `TI_PIN_MODE`（附录 B-4） |
| `TI_CLAIM_MLOCK_GB` | budget.py:766 | 未设 | mlock 上限 | 同上 |
| `TI_READERS` | engine.py:673 | "0" → 自动 | 覆盖并行读深度 | 与 `TI_READ_THREADS` 合并 |
| `TI_READ_THREADS` | loading.py:1070、1785；resources.py:829 | 未设 | 覆盖读线程数 | 同上 |
| `TI_MAX_BATCH` | sched.py:247、server.py:1594 | 未设 → 算 | 覆盖批宽 | 保留（压测固定） |
| `TI_BF16_CACHE` | deepseek/model.py:106 | 关 | 开：量化时也缓存 8 个 bf16 专家 | 删（量化/非量化混存，语义可疑） |
| `TI_THINKING_MODE` | server.py:80 | "chat" | chat / thinking | 保留（产品功能，不是优化开关） |

### A-2 仅做调试打印，不改变计算路径（约 30 个）

`TI_GPUDBG`（12 处）、`TI_DEBUG_ATTN`、`TI_DEBUG_IDX`、`TI_DEBUG_FLAT`、`TI_DEBUG_NAN`、`TI_FLAT_DBG`、`TI_MEM_DBG`、`TI_MEM_TRACE`、`TI_LAYER_TIME`、`TI_TIME`、`TI_SPEC_DEBUG`、`TI_SLOT_DEBUG`、`TI_CACHE_TRACE`、`TI_KV_TRACE`、`TI_GRAPH_TRACE`、`TI_WALL_PROF`、`TI_ATTN_PROF`、`TI_LINDBG`、`TI_FP8_DEBUG`、`TI_HEAD_DEBUG`、`TI_MOE_PROF`、`TI_MOE_CHK`、`TI_MARLIN_DEBUG`、`TI_EP_DBG`、`TI_HC_T`、`TI_FUSE_SYNC`。

**统一建议**：全部改成 `logging` 的 DEBUG 级别。现状是用 `print(..., flush=True)` 散落 60 多处，且每次调用都 `environ.get` 一次系统调用。删掉这些环境变量名之后，「这个项目到底有多少个开关」才会有一个可回答的答案。特别注意 `TI_MARLIN_DEBUG`（gemv_fp4_marlin.py:131、299）用的是 `!= "0"` —— **默认就在打印**，只是没人看日志。

### A-3 幽灵（无读取，11 个）

见 P0-B-1 表格。

### A-4 测试 / 工具专用（不必进运行时）

`TI_MODEL`（kcheck/*）、`TI_FUSE`（**import 期只读一次**，model.py:21）、`TI_HC_NOFUSE`（幽灵）、`TI_STREAM_CHECK`（kernels/native.py:257）、`TI_CALIB_DIR`（calibrate.py:31）、`TI_DETERMINISTIC`（engine.py:224）、`TI_TORCH_ONLY`（engine.py:227）、`TI_YARN_MSCALE`（nn.py:458）、`TI_DERIVED_MB`（compressed_attention.py:166）、`TI_SINKHORN`（hc.py:235）、`TI_HC_KERNEL_SOFT`（hc.py:510）、`TI_MARLIN_PYD`（gemv_fp4_marlin.py:315）、`TI_IOCP`（loading.py:3931）、`TI_PIN_CACHE_GB`（loading.py:3859，**在死函数 `_pin_cap` 里**）。

---

## 附录 B — 确实有运行时理由的开关（为什么不能自动检测代替）

判定标准：**同一次部署、同一台机器上，两条路径都可能更优，且差异来自外部状态而非硬件能力。**

### B-1 `TI_DIRECT`（engine.py:715）—— 保留

- 开：`NO_BUFFERING` 直读绕开页缓存；关：走页缓存。
- **为什么不能自动检测**：它取决于（a）文件系统与驱动（Windows NO_BUFFERING vs Linux ext4 pread）、（b）文件布局、（c）**页缓存的当前温度** —— 也就是「这次进程之前有没有人读过这批权重」。engine.py:699-701 记录的实测是 `TI_DIRECT=1` 时 2647ms/step、读 2.59GB，`TI_DIRECT=0` 时 2508ms/step、读 0.36GB —— **同一台机器上两者差不到 5%**，谁赢取决于当时页缓存里有多少。自动探测本身会把页缓存加热到让结论失效。
- **可接受的做法**：交给 `calibrate.py`（engine.py:307-316 已有启动期实测机制）测一次写进 `tuned.json`，代码里不要再留运行时分支。**这是「用标定代替开关」，不是「保留开关」。**

### B-2 `TI_SLAB_FRAC`（engine.py:2217，默认 0.28）—— 保留但应移入规划器

- 语义：pin 池里分给「层 slab」的比例上限，剩下给专家缓存。
- **为什么不能自动检测**：这是一个**互相争抢的静态分割**，最优值随模型的 `n_layers`、每层字节、专家缓存命中率三者联动变化，而命中率本身随 workload 变化。engine.py:2214-2215 记录「命中率 0.66→0.75 让 step 从 574ms 降到 519ms」，即缓存比 slab 更值钱；但这个结论只在这台机器这个模型上成立。
- **可接受的做法**：同 B-1，交给标定 + 规划器（`resources.plan` 已在算 `cache_bytes` 与 slab，这里的 env 是第二套算法）。

### B-3 `TI_NO_SLOTS`（engine.py:786）—— 保留，但要改名，而且它背后是一个 bug

- 语义：关掉显存专家槽位，换可复现的 logits。engine.py:738-740 原文：「The VRAM slot pool is faster but not bit-reproducible yet（a slot can be re-targeted while the GEMV that reads it is still queued）…… `TI_NO_SLOTS=1` trades about 40% of decode for deterministic logits.」
- **为什么不能自动检测**：这是**用户对正确性的要求**，不是硬件属性。做数值对拍、复现 bug、跑 CI 时必须关槽位；生产要速度。
- **建议**：改名为 `TI_DETERMINISTIC_DECODE` 并与现有 `TI_DETERMINISTIC`（engine.py:224）合并 —— 现在两个「确定性」开关做两件不同的事，名字却分不出来。**同时要清楚：槽位被正在执行的 GEMV 读到时重新指向，是正确性缺陷，不是配置选项。** 修好（给槽位加 ref/event 屏障）之后这个开关就该消失。

### B-4 `TI_CLAIM_TOUCH` / `TI_CLAIM_MLOCK_GB`（budget.py:751、766）—— 保留，应合三为一

- 语义：claim 到 pin 池后是否立即 touch 锁页、以及 mlock 上限。
- **为什么不能自动检测**：取决于宿主机的 `RLIMIT_MEMLOCK`（Linux 上常见几十 MB 到几 GB）、是否有 root、以及「现在要不要把启动时间换成确定性」。budget.py:763-765 明确记录「280GB 必然失败，只是失败之前已经白干了」。启动时先 try 再回退是可以的，但会付出一次全量 touch 的时间（几百 GB 的页）。
- **建议**：合并成一个 `TI_PIN_MODE=lazy|touch|mlock` 三态，而不是三个独立变量 + 一个幽灵 `TI_CLAIM_REAL`。

### B-5 `TI_RING`（engine.py:2179，默认 3）—— 删

- 语义：层权重环形缓冲深度。engine.py:2177-2178：「depth 2 = 1.04-1.27s/token = 0.867。Deeper is not free（每个额外 ring slot 占 463MB of the 6.9GB card），so 3 is the measured optimum, not a maximum.」
- **为什么不需要开关**：它已经可以自动检测 —— `_shed_vram`（engine.py:3026）在 OOM 时就在收缩 ring。让 `_shed_vram` 的单向收缩成为唯一路径，default 3 保留为代码常量即可。
- 这条是「看起来像 B 类、实际是 A 类」的例子：注释给了实测理由，但代码里已经有自适应机制。

### B-6 `TI_HEAD_CPU`（resources.py:529）—— 删，属于「调试价值不是保留理由」

- 唯一的「理由」是 resources.py:532-534 自己写的：「`_head_logits` 没有 host-GEMV 路径（它在设备上切片 head），所以 `head_on_cpu` 的规划只会表现为后续 device mismatch。**保留 head 在卡上，除非操作者显式强制 TI_HEAD_CPU**。」
- 也就是说：这个开关打开必然导致设备不匹配，它只用于「制造那个错误」。这正是任务里说的「不要因为可能有调试价值就保留」。

### 不需要保留、也不属于以上任何一类的

`TI_ZC`（地址算术自认不可信）、`TI_V5`（入口函数已死）、`TI_ALL_PINNED`（纯测量）、`TI_EP`（自认答案会错）、`TI_GPU_LAYERS` / `TI_RESIDENT` / `TI_MAX_RESIDENT`（注释都写着 for experiments / forced）、`TI_GRAPH=1` 分支（30% 500 率）。这些是实验脚手架，不是运行时选择。

---

## 附录 C — 审计方法与局限

**做了什么**：

1. 全量枚举 `TI_*`（唯一名 104、引用 219 处），逐条取 ±2 行上下文判读默认值与分支走向。
2. 用「±1 行窗口内是否有 `environ.get`」机械核对每个变量是否真的被读，再人工复核 → 得到 11 个幽灵开关。
3. `ast.parse` 全量扫描：常量 False 的 `if` / `while`（1 处）、`return` / `raise` / `continue` / `break` 之后同层的语句（1 处）。
4. 提取 1268 个 `def`，统计每个的「全包内引用数」，剔除 dunder、property pair、装饰器注册的 handler 后得到 108 个零引用定义，再逐个回读确认。
5. 模块级 import 图（92 × 92）找出 6 个零 import 孤儿。
6. `ast` 扫描「同一模块内 def 重名」与「同一类内方法重名」。
7. 对 `TierConfig` 的 22 个字段逐一核对有无读者（含 `getattr(tier, "字段名")` 形式的字符串读取）。

**没做**（因此以下结论保持「怀疑」）：

- 没有加载模型、没有跑 benchmark。所有性能数字都是**代码注释里已有的实测值**，我没有复现。凡引用处我都标了行号，可逐条核对。
- 没有运行时覆盖率验证。P2 的「零引用」是**静态**结论：`getattr(obj, "name")`、字符串注册表、外部插件调用这三类不会被静态扫描捕获。`server.py` 的 pydantic validator（`_check_stop` / `_content_parts` / `_check_messages`）已确认属此类误报并标注；其余条目若未来接入 `getattr` 式调用，需重新核对。
- 没有审计 `kernels/cuda/*.cu` 里的 `#if 0` / 未启用 kernel（本次范围是 `src/tinyinfer` 下的 Python）。`kernels/build/*.cubin` 里有 40+ 个已编译 kernel，是否全部有 Python 侧调用点值得单独做一次（`kcheck/` 覆盖了一部分）。
- 没有核对 `docs/*.md` 与代码的一致性（只在 `docs/设计-专家加载.md` 里发现 4 处引用了死函数：416 / 419 / 420 / 422 行）。

**清理顺序建议**（按「删掉后风险最小、收益最大」排）：

1. 先删 P2-4 六个孤儿模块与 P2 里 engine / loading 的零引用函数 —— 纯删除，无行为变化。
2. 再删 P0-B-1 那 11 个幽灵开关的**注释**（代码本来就没有），以及 tiers.py 里对应的 notes —— 只改文本，但消除了「以为能用」的错觉。
3. 然后处理 P0-5 / P0-6 / P0-7 的「只写不读字段」：要么接上，要么字段与写入点一起删。**这一步之后，tier 配置才第一次与运行行为一一对应。**
4. 再处理 P0-3 / P0-4：删 `TI_SERVE_LOOP` 前必须先把 prefix 复用迁到批处理路径（这是一个功能，不是一个开关）。
5. P0-1 / P0-2 / P0-8 / P0-10 需要真实测量才能定「删掉还是接上」，建议各自单独立项，不要塞进同一个改动。
6. 最后做 P3 的合并（4 套预取 → 1、5 套槽位 → 1、2 套调度器 → 1）。这一步最大，但它才是「落后架构不存在」的实质内容。

第 1、2 步合起来可以让 `TI_*` 从 104 个降到 90 个左右，且**不改变任何运行时行为** —— 是风险最低的第一刀。
