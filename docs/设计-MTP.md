# 设计：MTP（多 token 预测）

## 定位

主干之后挂 3 个 MTP stage，用于投机解码。它们不在普通前向里跑，只在开启推测时进入。
是否启用由规划器决定——它要占显存，装不下就不开。

## 与官方实现的对应

迁移或修复时对照官方 `mtp/dflash.py`。字段名在两边的差异：

| 本引擎 | 官方 | 说明 |
| --- | --- | --- |
| `MTPBlock.main_proj` / `main_norm` | `projection` / `project_norm` | target 层 hidden 的投影与归一 |
| `MTPBlock.main_hiddens` | `DFlash.update_main_cache` | 拼 target 层 hidden 得到 main_x |
| `MTPBlock.forward_embed` | `DFlash._embed_draft_init_tokens` | noise 填充 + 首 token |
| `MTPBlock.forward` | `DFlashDecoderLayer.forward_drafts` | HC + attention + MoE |
| `MTPAttention.forward_with_main` | `DFlashAttention` | main 滑窗 + draft KV |
| `MTPBlock.forward_head` | `DFlash.sample_drafts` | post_norm → head → markov → confidence |
| `markov_embed` / `markov_head` | `shared_head.markov_w1` / `markov_w2` | |
| `confidence_proj` | `confidence_head.proj` | |

vLLM 的 nextn 层（`num_nextn_predict_layers=3`）与此同构。参数：draft block_size=5，
markov_rank=256。

## 权重名

```
mtp.{S}.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight / .scale      fp8
mtp.{S}.attn.{q_norm,kv_norm}.weight, attn_sink
mtp.{S}.attn_norm.weight / ffn_norm.weight
mtp.{S}.hc_attn_fn / hc_attn_scale / hc_attn_base / hc_ffn_*
mtp.{S}.ffn.gate.{weight,bias,bias_vl} / shared_experts.* / experts.{i}.{w1,w2,w3}.{weight,scale}
mtp.0.main_proj.weight / .scale     [dim, n_target*dim]
mtp.0.main_norm.weight              [dim]
mtp.2.norm.weight                   [dim]
mtp.2.markov_head.embed.weight      [vocab, 256]
mtp.2.markov_head.head.weight       [256, vocab]
mtp.2.confidence_head.proj.weight   [1, dim+256]
```

## 契约

- **main_x** = `projection_norm(main_proj(cat(h_L37, h_L38, h_L39)))`。三层 hidden 都
  取 **HC 折叠之后**的 `[1, T, dim]`。
- **main KV** = `kv_norm(wkv(main_x))`，rope 之后进 main 滑窗，窗口 128。每个 stage
  用自己的 wkv。
- **draft 输入**：主 embedding 的 noise 填充块（`noise_token_id = 128799`），位置 0
  放 golden token。
- **draft KV**：每 stage 的 `wkv(x)`，rope 之后**只在本次 block 内有效**。
- **attention**：把 `[main 窗口, draft KV]` 拼起来送稀疏注意力（sink + 窗口索引）。
- **head**：`post_norm(hc_pre(x))` → 主 head logits → markov 修正
  （`embed(ids) @ markov_head`）→ confidence
  （`sigmoid(proj([hidden, markov_emb]))`）。
- **位置**：draft 位置 = `start_pos .. start_pos+4`；main 位置 = `start_pos - 1`。

## 投机解码流程

1. 主模型前向得到 logits，贪心采样得 t0，并捕获 target 层的 hidden。
2. `main_x = stage0.main_hiddens(target_hiddens)`；`x0 = stage0.forward_embed(t0)`。
3. stage0 → stage1 → stage2 依次 `forward`；最后一个 stage 的 `forward_head` 产出
   draft tokens 及置信度。
4. 主模型一次并行前向做 verify，贪心匹配接受前缀；不匹配则用 snapshot/restore 回滚 KV。
5. 接受率与加速比记录进基准。

## 当前状态

权重加载与 stage0/1/2 的前向链（attention + MoE + HC）已实现。
`forward_head`（markov 与 confidence 部分）尚未完成，因此投机解码还不会带来加速。
