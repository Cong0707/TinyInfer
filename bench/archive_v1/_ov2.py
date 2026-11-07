read_ms = 442.0
total = 1065.0
compute = total - read_ms
print(f'RESULT read {read_ms:.0f} ms, compute {compute:.0f} ms')
print(f'RESULT 完全重叠需要 compute >= read: {compute >= read_ms}')
print(f'RESULT 若完全重叠, step = {max(read_ms, compute):.0f} ms')
print(f'RESULT 收益: {(1-max(read_ms,compute)/total)*100:.0f}%')
# 但 40 层串行，每层 read ~11ms、compute ~15.6ms
print()
print(f'RESULT 每层: read {read_ms/40:.1f} ms, compute {compute/40:.1f} ms')
print('RESULT 若层间能流水: 首层等 read, 之后取 max(11, 15.6) = 15.6 ms/层')
print(f'RESULT 理想 step = {read_ms/40 + 39*15.6:.0f} ms  (当前 {total:.0f})')