data = [(2,0.118,79.23),(4,0.235,72.40),(8,0.484,55.88),(16,0.683,38.07),(24,0.685,37.96)]
print('RESULT GB   命中率  磁盘GB   边际增益(每GB)')
prev = None
for gb, hit, disk in data:
    if prev:
        dg = gb - prev[0]; dh = hit - prev[1]; dd = prev[2] - disk
        print(f'RESULT {gb:3d}  {hit:6.3f}  {disk:7.2f}   +{dh/dg*100:.1f}pp/GB, -{dd/dg:.2f} GB 磁盘/GB')
    else:
        print(f'RESULT {gb:3d}  {hit:6.3f}  {disk:7.2f}')
    prev = (gb, hit, disk)
print()
print('RESULT === 换算成时间（3.0 GB/s，8 步）===')
for gb, hit, disk in data:
    ms = disk/3.0/8*1000
    print(f'RESULT {gb:3d} GB: 磁盘 {ms:4.0f} ms/步 -> {1000/(ms+120):.2f} tok/s（粗算，+120ms 非磁盘部分）')