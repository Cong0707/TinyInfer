base = 3664
ring = 3*463
w10 = 1640
ws = 500
used = w10 + base + ring + ws
print('RESULT 8GB卡占用: 常驻10层 %d + base %d + ring %d + ws %d = %d MB (%.2f GB / 6900)' % (w10, base, ring, ws, used, used/1024))
print('RESULT 剩余 %d MB' % (6900-used))
print('RESULT 30层需要 %d MB (%.2f GB)' % (30*165.9, 30*165.9/1024))
print('RESULT 还能再放 %.1f 层' % ((6900-used)/165.9))
print('RESULT 40层全部常驻需要 %.0f MB = %.2f GB' % (40*165.9, 40*165.9/1024))
print('RESULT 24GB卡可全常驻 -> H2D 只传专家 4.51 GB (省61%%)')