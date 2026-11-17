"""索引器使用 attention 的共享位置表，不把它当层权重搬运。"""
from types import SimpleNamespace

import torch

from tinyinfer.deepseek.model import Block
from tinyinfer.deepseek.compressed_attention import MLACompressedAttention


def test_weight_paths_exclude_both_rope_table_references():
    weight = torch.ones(1)
    frequencies = torch.zeros(256, dtype=torch.complex64)
    block = Block.__new__(Block)
    for name in ("attn_norm", "ffn_norm", "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                 "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
        setattr(block, name, weight)
    indexer = SimpleNamespace(wq_b=weight, weights_proj=weight, wk=weight,
                              k_norm=weight, freqs_cis=frequencies)
    block.attn = SimpleNamespace(q_norm=weight, kv_norm=weight, sink=weight,
                                 wq_a=weight, wq_b=weight, wkv=weight,
                                 wo_a=weight, wo_b=weight, freqs_cis=frequencies,
                                 indexer=indexer, compressor=None)
    block.ffn = SimpleNamespace(gate=SimpleNamespace(weight=weight),
                                shared=SimpleNamespace(w1=weight, w2=weight, w3=weight))
    paths = block._weight_paths()
    assert not any(name == "freqs_cis" for _, name, _ in paths)
    assert any(owner is indexer and name == "wq_b" for owner, name, _ in paths)
    assert block.attn.freqs_cis is block.attn.indexer.freqs_cis


def test_attention_migration_resolves_one_shared_table(monkeypatch):
    import tinyinfer.deepseek.compressed_attention as module

    attention = MLACompressedAttention.__new__(MLACompressedAttention)
    weight = torch.ones(1)
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "q_norm", "kv_norm", "sink"):
        setattr(attention, name, weight)
    attention._rope_key = ("test",)
    attention._rope_factory = lambda: None
    attention.compressor = None
    attention.indexer = SimpleNamespace(freqs_cis=object())
    seen = []
    attention.indexer.to = lambda device: seen.append(attention.indexer.freqs_cis)
    table = torch.zeros(16, dtype=torch.complex64)
    monkeypatch.setattr(module, "get_table", lambda *args: table)
    attention.to("cpu")
    assert attention.freqs_cis is attention.indexer.freqs_cis is table
    assert seen == [table]
