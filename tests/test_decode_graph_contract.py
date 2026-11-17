"""图失败必须中止，不得静默执行第二次前向或缩减布局。"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tinyinfer.engine import Engine


def make_engine():
    eng = Engine.__new__(Engine)
    eng.expert_store = SimpleNamespace(slots=SimpleNamespace(_slot_tab=object()))
    eng._graph_run = Mock(return_value=None)
    eng._forward_impl = Mock()
    eng._graph_try_capture = Mock()
    return eng


@pytest.mark.parametrize("error", [RuntimeError("capture failed"), torch.OutOfMemoryError("out of memory")])
def test_capture_failure_does_not_execute_another_forward(error):
    eng = make_engine()
    eng._graph_try_capture.side_effect = error
    with pytest.raises(type(error)) as caught:
        eng._forward_inner(torch.tensor([[1]]), 3)
    assert caught.value is error
    eng._graph_try_capture.assert_called_once()
    eng._forward_impl.assert_not_called()


def test_prefill_oom_is_not_retried_with_smaller_layout():
    eng = make_engine()
    eng._forward_impl.side_effect = torch.OutOfMemoryError("out of memory")
    with pytest.raises(torch.OutOfMemoryError):
        eng._forward_inner(torch.tensor([[1, 2]]), 0)
    eng._forward_impl.assert_called_once()
    assert not hasattr(eng, "_shed_vram")


def test_graph_hit_does_not_enter_capture_or_direct_forward():
    eng = make_engine()
    output = torch.ones(1)
    eng._graph_run.return_value = output
    assert eng._forward_inner(torch.tensor([[1]]), 3) is output
    eng._graph_try_capture.assert_not_called()
    eng._forward_impl.assert_not_called()


def test_initial_forward_builds_table_once():
    eng = make_engine()
    eng.expert_store.slots._slot_tab = None
    ids = torch.tensor([[1]])
    eng._forward_inner(ids, 0)
    eng._forward_impl.assert_called_once_with(ids, 0, None, positions=None)
    eng._graph_try_capture.assert_not_called()


def test_whole_graph_rejects_ring_reuse_before_warmup():
    eng = Engine.__new__(Engine)
    eng._graph_capture_shape = Mock(return_value=True)
    eng._stream_layers = Mock(return_value={1, 2, 3})
    eng._flat_state = ({}, [object(), object()], {})
    eng._forward_impl = Mock()
    eng.snapshot = Mock()
    ids = torch.tensor([[1]])
    with pytest.raises(RuntimeError, match="weight-lifetime segments"):
        eng._graph_try_capture(ids, 3)
    eng._forward_impl.assert_not_called()
    eng.snapshot.assert_not_called()
    with pytest.raises(RuntimeError, match="previous CUDA graph capture failed"):
        eng._graph_try_capture(ids, 3)
    eng._graph_capture_shape.assert_called_once()


def test_failed_forward_is_not_a_layout_observation():
    eng = make_engine()
    eng._planner = SimpleNamespace(observe=Mock())
    eng._forward_inner = Mock(side_effect=RuntimeError("failed step"))
    with pytest.raises(RuntimeError, match="failed step"):
        eng.forward(torch.tensor([[1]]), 3)
    eng._planner.observe.assert_not_called()
    assert not getattr(eng, "_observed_first", False)
