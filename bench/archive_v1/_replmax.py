import io
p = r"E:\Develop\Deepseek\tinyinfer\repl.py"
t = io.open(p, encoding="utf-8").read()

# `-n 64` was a debug default that truncated every answer mid-sentence; the server already
# stops at EOS, so the client cap only needs to be a safety bound.
old = "    max_tokens = 64"
new = "    max_tokens = 2048"
assert old in t
t = t.replace(old, new, 1)

old2 = 'def chat(messages, max_tokens: int = 64, temperature: float = 0.0,'
new2 = 'def chat(messages, max_tokens: int = 2048, temperature: float = 0.0,'
assert old2 in t
t = t.replace(old2, new2, 1)

old3 = '            print("  -n, --max-tokens N    cap on generated tokens (default 64)")'
new3 = '            print("  -n, --max-tokens N    hard cap on generated tokens (default 2048;")' \
       '\n                  print("                        generation stops at EOS long before that)")'
if old3 in t:
    t = t.replace(old3, new3, 1)

io.open(p, "w", encoding="utf-8", newline="").write(t)
print("ok")