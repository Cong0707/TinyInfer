import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\hf_graph.py"
s = io.open(p, encoding="utf-8").read()
old = """                    self.graph = g
                    self.fallback_reason = None
                    self.capture_token = first_out
                    self.pos_buf.fill_(pos0 + 1)
                    self.ids_buf.fill_(first_out)
                    torch.cuda.synchronize()
                    return self"""
new = """                    self.graph = g
                    self.fallback_reason = None
                    self.capture_token = first_out
                    self.pos_buf.fill_(pos0 + 1)
                    self.ids_buf.fill_(first_out)
                    # The state replay expects is *after* this step, which is what the
                    # cache holds now. Keep a copy so a later request can rewind instead
                    # of capturing again (a capture costs ~137ms).
                    self.ready_state = _cache_snapshot(self.cache)
                    self.ready_pos = pos0 + 1
                    self.ready_token = first_out
                    torch.cuda.synchronize()
                    return self"""
assert old in s, "capture tail missing"
s = s.replace(old, new, 1)

anchor = "    def replay_step(self) -> None:"
add = '''    def rewind(self) -> None:
        """Put the cache back to the state the capture left it in, ready for replay.

        A capture runs the step to prove the graph matches eager, so the cache ends one
        token ahead with that token sampled. Replaying from there is correct once; a second
        request must start from the same point, and re-running capture would cost ~137ms
        per completion.
        """
        state = getattr(self, "ready_state", None)
        if state is None:
            return
        with torch.inference_mode():
            _cache_restore(state)
            self.pos_buf.fill_(self.ready_pos)
            self.ids_buf.fill_(self.ready_token)

'''
assert "def rewind" not in s
i = s.index(anchor)
s = s[:i] + add + s[i:]
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("rewind added")