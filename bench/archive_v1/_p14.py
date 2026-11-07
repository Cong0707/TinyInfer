import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\hf_graph.py"
s = io.open(p, encoding="utf-8").read()
old = """        state = getattr(self, "ready_state", None)
        if state is None:
            return
        with torch.inference_mode():
            _cache_restore(state)
            self.pos_buf.fill_(self.ready_pos)
            self.ids_buf.fill_(self.ready_token)"""
new = """        state = getattr(self, "ready_state", None)
        if state is None:
            return
        with torch.inference_mode():
            # ready_state is the cache *before* the step the capture consumed, paired with
            # the token that step read. Restoring both and leaving pos at pre-step means the
            # next replay executes that same step, so its output is produced rather than
            # skipped -- which is what a fresh decoder emits via drain().
            _cache_restore(state)
            self.pos_buf.fill_(self.ready_pos)
            self.ids_buf.fill_(self.ready_token)"""
assert old in s
s = s.replace(old, new, 1)

# capture must therefore store the PRE-step state and its input token
old2 = """                    # The state replay expects is *after* this step, which is what the
                    # cache holds now. Keep a copy so a later request can rewind instead
                    # of capturing again (a capture costs ~137ms).
                    self.ready_state = _cache_snapshot(self.cache)
                    self.ready_pos = pos0 + 1
                    self.ready_token = first_out"""
new2 = """                    # Keep the state the step started from, plus the token it read. A
                    # later request rewinds to exactly this point and replays the step, so
                    # it reproduces the same first token instead of skipping it, and no
                    # second capture is needed (one costs ~137ms).
                    self.ready_state = snap
                    self.ready_pos = pos0
                    self.ready_token = tok0"""
assert old2 in s
s = s.replace(old2, new2, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")