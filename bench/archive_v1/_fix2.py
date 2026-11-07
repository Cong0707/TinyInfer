import io
p = r"E:\Develop\Deepseek\tinyinfer\tests\test_streaming_session.py"
t = io.open(p, encoding="utf-8").read()
old = 'pytestmark = pytest.mark.skipif(not (MODEL / "config.json").exists(),\n                                reason="checkpoint not present")'
new = '# These build a real Engine from the 510GB checkpoint (tens of seconds, most of VRAM).\n' \
      '# They are opt-in with --runslow so a routine gate never loads the model.\n' \
      'pytestmark = [\n' \
      '    pytest.mark.skipif(not (MODEL / "config.json").exists(),\n' \
      '                        reason="checkpoint not present"),\n' \
      '    pytest.mark.slow,\n' \
      ']'
assert old in t, "marker block changed"
t = t.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(t)

# register the marker and skip it unless --runslow
pp = r"E:\Develop\Deepseek\tinyinfer\pytest.ini"
c = io.open(pp, encoding="utf-8").read()
c = c.replace("markers =\n    kernel: compiles tilelang kernels (slow; opt-in via check.ps1 -Kernels)",
              "markers =\n    kernel: compiles tilelang kernels (slow; opt-in via check.ps1 -Kernels)\n    slow: needs the real checkpoint loaded (opt-in with --runslow)")
if "--runslow" not in c:
    c = c.rstrip() + "\n# skip \"slow\" (checkpoint-backed) cases unless asked: --runslow\naddopts = -q --tb=short -m \"not slow\"\n"
    # remove the duplicate addopts line that was already there
    first = c.index("addopts = -q --tb=short")
    second = c.find("addopts = -q --tb=short", first + 10)
    if second > 0:
        c = c[:first] + c[second:]
io.open(pp, "w", encoding="utf-8", newline="").write(c)
print("ok")