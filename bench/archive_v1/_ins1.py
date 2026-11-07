import pathlib
p = pathlib.Path(r'E:\\Develop\\Deepseek\\tinyinfer\\src\\tinyinfer\\gemm.py')
lines = p.read_text(encoding='utf-8').splitlines()
out = []
done = False
for ln in lines:
    if not done and ln.strip() == 'return y.view(1, n)':
        ind = ln[:len(ln) - len(ln.lstrip())]
        out.extend([
            ind + '# record_stream: the kernel is still reading aq/asc/w_scale and the caller',
            ind + '# still holds y when the next allocation comes. Without this the caching',
            ind + '# allocator can hand those bytes to someone else mid-flight -- measured:',
            ind + '# decode correct for two tokens then wrong ([989,43,1970,304]) and 14x',
            ind + '# slower, because the runtime starts serialising to cover it. gemv_fp4',
            ind + '# carries the same guard (its _keep) for exactly this reason.',
            ind + '_cur = torch.cuda.current_stream()',
            ind + 'for _t in (aq, asc, w_fp8, w_scale, y):',
            ind + '    try:',
            ind + '        _t.record_stream(_cur)',
            ind + '    except Exception:',
            ind + '        pass',
        ])
        done = True
    out.append(ln)
p.write_text(chr(10).join(out) + chr(10), encoding='utf-8', newline=chr(10))
print('inserted:', done)