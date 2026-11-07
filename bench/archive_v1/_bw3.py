BPT = 11.02
cards = [
    ('RTX 5060 Laptop (8GB, this box)', 384, 1.26),
    ('RTX 5060 (desktop)', 448, None),
    ('RTX 2080 Ti (11GB)', 616, None),
    ('RTX 5090 (32GB)', 1792, None),
    ('RTX PRO 6000 Blackwell (96GB)', 1792, None),
    ('H100 SXM (80GB HBM3)', 3350, None),
    ('H100 NVL (94GB HBM3)', 3900, None),
    ('H200 SXM/NVL (141GB HBM3e)', 4800, None),
    ('B300 SXM (288GB HBM3e)', 8000, None),
]
print('RESULT %-34s %8s %10s %10s %8s' % ('card', 'GB/s', 'ceiling', 'measured', 'share'))
print('RESULT ' + '-'*76)
for name, bw, meas in cards:
    ceil = bw / BPT
    ms = 1000.0 / ceil
    m = '%.1f' % meas if meas else '--'
    sh = '%.1f%%' % (meas/ceil*100) if meas else '--'
    print('RESULT %-34s %8d %8.1f %10s %8s' % (name, bw, ceil, m, sh))
print()
print('RESULT === 300 tok/s 需要 3307 GB/s ===')
for name, bw, _ in cards:
    if bw >= 3307:
        print('  %-34s %d GB/s -> 100%% 时 %.1f tok/s  [可达]' % (name, bw, bw/BPT))
print('  其余全部不足')
print()
print('RESULT === B300 的余量 ===')
print('  8000/3307 = %.2fx 余量' % (8000/3307))
print('  即 kernel 只需达到 %.0f%% 带宽就能跑 300 tok/s' % (3307/8000*100))
print()
print('RESULT === H200 呢 ===')
print('  4800 GB/s -> 100%% 时 %.1f tok/s' % (4800/BPT))
print('  要到 300 需 %.0f%% 效率' % (3307/4800*100))
print()
print('RESULT === 但 288GB 装不下 475GB 模型 ===')
print('  B300 288GB < 475.3GB -> 仍需 2 张 -> 每张要跑一半')
print('  2 张 B300 = 16000 GB/s, 模型 475GB 分成两半各 238GB 放得下')
print('  但 MoE 层间依赖需要 AllReduce/AllGather')
print('  单 token 计算 11.02GB 分布在两卡，各搬 5.51GB -> 5.51/8 = 0.69ms')
print('  => 2 张 B300 理论 1.38ms = 725 tok/s，但通信开销是新的变量')