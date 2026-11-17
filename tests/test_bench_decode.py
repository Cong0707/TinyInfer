"""基准本身必须测连续解码，不能把 reset/重复 prefill 当 decode。"""
import importlib.util
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "bench_decode", Path(__file__).resolve().parents[1] / "scripts" / "bench_decode.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


class Engine:
    def __init__(self, invalid_at=None):
        self.calls = []
        self.invalid_at = invalid_at

    def forward(self, ids, position):
        self.calls.append((ids.clone(), position))
        logits = torch.zeros((1, ids.shape[1], 32))
        logits[:, -1, len(self.calls)] = float("nan") if position == self.invalid_at else 1.0
        return logits


def test_positions_and_generated_tokens_are_continuous():
    eng = Engine()
    rows = []
    result = bench.measure_decode(
        eng, torch.tensor([[4, 5, 6]]), tokens=6, warmup=2,
        synchronize=lambda: None, emit=rows.append)
    assert [p for _, p in eng.calls] == [0, 3, 4, 5, 6, 7, 8, 9, 10]
    assert [ids.item() for ids, _ in eng.calls[1:]] == list(range(1, 9))
    assert [r["phase"] for r in rows] == ["prefill"] + ["warmup"] * 2 + ["decode"] * 6
    assert result["n"] == 6
    assert result["finite"]


@pytest.mark.parametrize("invalid_at", [0, 3, 5])
def test_nonfinite_prefill_warmup_or_decode_never_produces_score(invalid_at):
    eng = Engine(invalid_at)
    rows = []
    with pytest.raises(FloatingPointError, match="nonfinite logits"):
        bench.measure_decode(eng, torch.tensor([[4, 5, 6]]), tokens=6, warmup=2,
                             synchronize=lambda: None, emit=rows.append)
    assert rows[-1]["finite"] is False
    assert rows[-1]["position"] == invalid_at


def test_invalid_sample_count_fails_before_forward():
    eng = Engine()
    with pytest.raises(ValueError):
        bench.measure_decode(eng, torch.tensor([[4]]), tokens=0, warmup=2,
                             synchronize=lambda: None, emit=lambda row: None)
    assert not eng.calls
