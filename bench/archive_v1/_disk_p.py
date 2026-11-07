base_disk_ms = 830.0
base_total = 790.0
print('RESULT 当前: 3.0 GB/s 磁盘, 2.49 GB/token, 790 ms')
print()
print('RESULT === 换不同速率的盘 ===')
for rate, name in ((3.0,'当前'), (5.0,'普通 NVMe'), (7.0,'高端 NVMe'), (14.0,'PCIe5 NVMe'), (100,'内存')):
    read_ms = 2.49/rate*1000
    # 无重叠，但计算 623ms 可部分覆盖
    total = max(read_ms, 623) + max(0, read_ms-623)*0
    # 更保守：串行相加再减去已知重叠
    sec = read_ms + max(0, 790-read_ms)
    print(f'  {name:10s} {rate:5.1f} GB/s -> 读 {read_ms:5.0f} ms -> 预计 {max(read_ms, 623):4.0f}-{read_ms+167:4.0f} ms = {1000/max(read_ms,623):.2f}-{1000/(read_ms+167):.2f} tok/s')
print()
print('RESULT 注：623ms 是实测的 compute 部分（无读取时的下限）')