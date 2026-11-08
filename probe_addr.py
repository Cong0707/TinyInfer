"""from_buffer 对 18.8MB memoryview 的取地址开销。"""
import ctypes, time

N = 18800640
buf = bytearray(N)
mv = memoryview(buf)

# 第一次
t0 = time.perf_counter()
for _ in range(20):
    a = ctypes.addressof(ctypes.c_char.from_buffer(mv))
dt = (time.perf_counter() - t0) / 20
print("from_buffer(memoryview) = %.3f ms each" % (dt * 1e3), flush=True)

# 换成直接对 bytearray
t0 = time.perf_counter()
for _ in range(20):
    a = ctypes.addressof(ctypes.c_char.from_buffer(buf))
dt2 = (time.perf_counter() - t0) / 20
print("from_buffer(bytearray)  = %.3f ms each" % (dt2 * 1e3), flush=True)

# 缓存地址
addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
t0 = time.perf_counter()
for _ in range(20):
    a = addr
dt3 = (time.perf_counter() - t0) / 20
print("cached address          = %.3f ms each" % (dt3 * 1e3), flush=True)
print("=> 20 次 DMA 浪费 %.1f ms" % ((dt - dt3) * 20 * 1e3), flush=True)
