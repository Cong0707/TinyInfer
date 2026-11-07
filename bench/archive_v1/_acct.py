disk = 672.0
ld = 370 * 16.5 / 1000
print(f'RESULT disk    : {disk:6.1f} ms  {disk/790*100:5.1f}%')
print(f'RESULT launch  : {ld:6.1f} ms  {ld/790*100:5.1f}%')
print(f'RESULT other   : {790-disk-ld:6.1f} ms')
print(f'RESULT if zero launches: {disk:.0f} ms = {1000/disk:.2f} tok/s (+{(1000/disk)/1.26-1:.1%})')
d2 = disk/2 + ld + 118
print(f'RESULT if disk halved  : {d2:.0f} ms = {1000/d2:.2f} tok/s (+{(1000/d2)/1.26-1:.0%})')