import pathlib
# 从 3.9 的 sm90 example 找可用的 stride 写法
base = pathlib.Path(r'E:\\Develop\\Deepseek\\repositories\\cutlass-3.9')
hits = []
for p in (base / 'examples').rglob('*.cu'):
    try: t = p.read_text(encoding='utf-8', errors='ignore')
    except Exception: continue
    if 'make_cute_packed_stride' in t and 'fp8_blockwise' in t.lower():
        hits.append(p)
        if len(hits) >= 3: break
for h in hits: print('EXAMPLE', h.relative_to(base))
if not hits:
    for p in (base / 'examples').rglob('*.cu'):
        try: t = p.read_text(encoding='utf-8', errors='ignore')
        except Exception: continue
        if 'make_cute_packed_stride' in t:
            print('STRIDE_EXAMPLE', p.relative_to(base))
            break