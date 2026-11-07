mt = 795.0; mf = 285.0; a = 0.114; b = 74.0; mb = 1852.0
def gain(newf):
    nt = 795 - (mf - newf)
    return nt, (1000/nt)/(1000/795) - 1
print('RESULT 1. 命中率 100%（磁盘 0）: fetch 285 -> %.0f ms' % b)
nt, g = gain(b); print(f'RESULT    total {nt:.0f} ms  {1000/nt:.2f} tok/s  {g:+.0%}')
f2 = a*1445+b
nt, g = gain(f2); print(f'RESULT 2. 磁盘 -22%（命中 +10pp）: fetch -> {f2:.0f} ms, total {nt:.0f} ms {g:+.1%}')
f3 = (a/2)*mb+b
nt, g = gain(f3); print(f'RESULT 3. 读速率翻倍: fetch -> {f3:.0f} ms, total {nt:.0f} ms {g:+.1%}')
print()
print('RESULT 命中率已 93%（30步），磁盘 3.0 GB/s 是物理上限 -> 三者都到顶')