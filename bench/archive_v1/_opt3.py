mt = 795.0; mf = 285.0; a = 0.114; b = 74.0; mb = 1852.0
def gain(newf):
    nt = 795 - (mf - newf)
    return nt, (1000/nt)/(1000/795) - 1
nt, g = gain(b)
print(f'RESULT 1 hit100: fetch {mf:.0f}->{b:.0f} ms, total {nt:.0f} ms, {1000/nt:.2f} tok/s, {g:+.0%}')
f2 = a*1445+b
nt, g = gain(f2)
print(f'RESULT 2 disk-22pct: fetch {mf:.0f}->{f2:.0f} ms, total {nt:.0f} ms, {g:+.1%}')
f3 = (a/2)*mb+b
nt, g = gain(f3)
print(f'RESULT 3 read2x: fetch {mf:.0f}->{f3:.0f} ms, total {nt:.0f} ms, {g:+.1%}')