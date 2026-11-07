import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\tinyinfer\\src\\tinyinfer\\gemm.py')
lines = p.read_text(encoding='utf-8').splitlines()
out = []
patched = 0
for i, ln in enumerate(lines):
    if ln.strip() == 'except Exception:' and i + 2 < len(lines) and lines[i + 2].strip() == 'pass':
        # 只在 fast path 那个 except（下一个非空行是 pass）后加打印
        if i > 0 and 'native.launch' in '\n'.join(lines[max(0,i-25):i]):
            ind = ln[:len(ln) - len(ln.lstrip())]
            out.append(ln)
            out.append(ind + '    import traceback as _tb')
            out.append(ind + '    print("[fp8-fast-fail]", repr(__import__("sys").exc_info()[1]), flush=True)')
            out.append(ind + '    _tb.print_exc()')
            patched += 1
            continue
    out.append(ln)
p.write_text(chr(10).join(out) + chr(10), encoding='utf-8', newline=chr(10))
print('patched', patched)