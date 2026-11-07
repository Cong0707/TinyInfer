import io
p = r"E:\Develop\Deepseek\tinyinfer\repl.py"
t = io.open(p, encoding="utf-8").read()

old = "def chat(messages, max_tokens: int = 64, temperature: float = 1.0,"
new = "def chat(messages, max_tokens: int = 64, temperature: float = 0.0,"
assert old in t
t = t.replace(old, new, 1)

# expose it on the command line
old2 = '    one_shot = " ".join(sys.argv[1:]).strip()'
new2 = """    argv = sys.argv[1:]
    temperature = 0.0
    max_tokens = 64
    rest: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] in ("-t", "--temperature") and i + 1 < len(argv):
            temperature = float(argv[i + 1]); i += 2; continue
        if argv[i] in ("-n", "--max-tokens") and i + 1 < len(argv):
            max_tokens = int(argv[i + 1]); i += 2; continue
        if argv[i] in ("-h", "--help"):
            print(__doc__.strip())
            print()
            print("  -t, --temperature F   sampling temperature (default 0 = greedy)")
            print("  -n, --max-tokens N    cap on generated tokens (default 64)")
            return 0
        rest.append(argv[i]); i += 1
    one_shot = " ".join(rest).strip()"""
assert old2 in t
t = t.replace(old2, new2, 1)

t = t.replace('        chat([{"role": "user", "content": one_shot}], session_id=sid)',
              '        chat([{"role": "user", "content": one_shot}], max_tokens=max_tokens,\n             temperature=temperature, session_id=sid)', 1)
t = t.replace("        reply = chat(messages, session_id=session)",
              "        reply = chat(messages, max_tokens=max_tokens,\n                    temperature=temperature, session_id=session)", 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)
print("repl patched")