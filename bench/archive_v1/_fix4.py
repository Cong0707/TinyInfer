import io
p = r"E:\Develop\Deepseek\tinyinfer\tests\test_server.py"
t = io.open(p, encoding="utf-8").read()
old = """    backend._bind_slot(3)
    assert engine.kv == {} and engine.resets == 1
    engine.kv[0] = sentinel_three
    backend._stash_slot(3)
    assert backend._slot_kv[3][0] is sentinel_three
    assert 4 not in backend._slot_kv

    backend._bind_slot(4)
    assert engine.kv == {}
    engine.kv[0] = sentinel_four
    backend._stash_slot(4)

    backend._bind_slot(3)
    assert engine.kv[0] is sentinel_three

    backend.reset_slot(3)
    assert 3 not in backend._slot_kv
    assert engine.kv == {}
    assert backend._slot_kv[4][0] is sentinel_four"""
new = """    # the slot map is keyed by session id, which falls back to the string "slot<N>" for
    # requests that carry no session -- the same key the on-disk .kvcache file uses
    k3, k4 = "slot3", "slot4"

    backend._bind_slot(3)
    assert engine.kv == {} and engine.resets == 1
    engine.kv[0] = sentinel_three
    backend._stash_slot(3)
    assert backend._slot_kv[k3][0] is sentinel_three
    assert k4 not in backend._slot_kv

    backend._bind_slot(4)
    assert engine.kv == {}
    engine.kv[0] = sentinel_four
    backend._stash_slot(4)

    backend._bind_slot(3)
    assert engine.kv[0] is sentinel_three

    backend.reset_slot(3)
    assert k3 not in backend._slot_kv
    assert engine.kv == {}
    assert backend._slot_kv[k4][0] is sentinel_four"""
assert old in t
t = t.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("ok")