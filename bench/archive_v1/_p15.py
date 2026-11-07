import io
p = r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\hf_graph.py"
s = io.open(p, encoding="utf-8").read()
# Reconstruct the state a fresh decoder is in when generate() starts: pos = pos0+1 and
# ids_buf = capture_token, i.e. the step that produced capture_token has been consumed.
# rewind() therefore restores the post-step cache (what the cache holds now) and leaves
# pos/ids at pos0+1 / first_out -- but then the next replay would repeat the step, so the
# fresh path is: emit capture_token, and replay from pos0+1.
old = """                    self.ready_state = snap
                    self.ready_pos = pos0
                    self.ready_token = tok0"""
new = """                    self.ready_state = _cache_snapshot(self.cache)
                    self.ready_pos = pos0 + 1
                    self.ready_token = first_out"""
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")