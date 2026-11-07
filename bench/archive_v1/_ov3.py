for total in (790, 1065):
    read_ms = 442.0 * (total/1065)
    compute = total - read_ms
    ideal = read_ms/40 + 39*max(read_ms/40, compute/40)
    print(f'RESULT total={total}: read={read_ms:.0f} compute={compute:.0f} ideal_pipelined={ideal:.0f} gain={100*(1-ideal/total):.0f}%')
print()
print('RESULT 注：read 442ms 是在 1065ms 那次测的，按比例缩放到 790ms 是 328ms')