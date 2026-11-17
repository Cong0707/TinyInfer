"""图外供货不能把缺失专家或尚未提交的传输当作 ready。"""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tinyinfer.expert_tiers import ExpertTierManager
from tinyinfer.loading import VramSlotCache


def manager():
    value = ExpertTierManager.__new__(ExpertTierManager)
    value.slots = VramSlotCache(2, 16, device="cpu")
    value.slots.ensure(3, [5, 7], lambda layer, wanted: {e: bytearray([e]) * 16 for e in wanted})
    value.slots.wait_ready = Mock()
    return value


def test_missing_requested_expert_is_an_error():
    value = manager()
    with pytest.raises(RuntimeError, match="not resident"):
        value.ready(3, [5, 9])
    value.slots.wait_ready.assert_not_called()


def test_complete_request_waits_once_per_slot():
    value = manager()
    lease = value.ready(3, [5, 7, 5])
    indices = [value.slots._by_key[(3, expert)] for expert in (5, 7)]
    value.slots.wait_ready.assert_called_once_with(indices)
    assert lease == [5, 7]
    assert value.slots._refs == [1, 1]
    value.release(3, lease)
    assert value.slots._refs == [0, 0]


def test_nonempty_request_requires_a_pool():
    value = manager()
    value.slots = None
    value.ready(3, [])
    with pytest.raises(RuntimeError, match="no slot pool"):
        value.ready(3, [5])


def test_staged_publication_does_not_discard_completion_event():
    cache = VramSlotCache(1, 16, device="cpu")
    cache._ensure()
    with cache._lock:
        assert cache._reserve_locked((0, 5)) == 0
    cache._state[0] = cache.COPYING
    event = object()
    cache.publish(0, event)
    assert cache._state[0] == cache.STAGED
    assert cache.event(0) is event


def test_reserved_slot_has_no_submitted_data():
    cache = VramSlotCache(1, 16, device="cpu")
    cache._torch = torch
    cache._state[0] = cache.RESERVED
    with pytest.raises(RuntimeError, match="not been submitted"):
        cache.wait_ready([0])


def test_copying_slot_requires_a_completion_event():
    cache = VramSlotCache(1, 16, device="cpu")
    cache._torch = torch
    cache._state[0] = cache.COPYING
    with pytest.raises(RuntimeError, match="without a completion event"):
        cache.wait_ready([0])


def test_pending_event_is_waited_without_an_optional_copy_stream_field():
    cache = VramSlotCache(1, 16, device="cpu")
    event = SimpleNamespace(query=lambda: False)
    consumer = SimpleNamespace(wait_event=Mock())
    cache._torch = SimpleNamespace(cuda=SimpleNamespace(
        device=lambda device: nullcontext(), current_stream=lambda: consumer))
    cache._buffers = [SimpleNamespace(device=torch.device("cuda:0"))]
    cache._state[0] = cache.STAGED
    cache._event[0] = event
    cache._copy_stream = None
    cache.wait_ready([0])
    consumer.wait_event.assert_called_once_with(event)
