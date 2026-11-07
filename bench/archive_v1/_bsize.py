print("=== Marlin B 的确切大小 ===")
print()
print("marlin_template.h:521")
print("  b_gl_stride = 16 * prob_n / (pack_factor * (is_a_8bit ? 2 : 4))")
print("  pack_factor = 32 / num_bits = 32/4 = 8  (int4)")
print("  is_a_8bit = false -> /4")
print("  => b_gl_stride = 16 * N / (8 * 4) = N / 2")
print()
N = 16384
K = 5120
b_gl_stride = 16 * N // (8 * 4)
print(f"N={N}: b_gl_stride = {b_gl_stride} int4 words")
print()
print("B 的总大小 = K/16 * b_gl_stride  (K/16 = k-tile 数)")
k_tiles = K // 16
total = k_tiles * b_gl_stride
print(f"  {k_tiles} x {b_gl_stride} = {total} int4 = {total*16/2**20:.1f} MB")
print()
print("理论上 fp4 权重应是 N*K/2 = " + f"{N*K/2/2**20:.1f} MB")
print(f"  实际需要 {total*16/2**20:.1f} MB")
print(f"  比例 {total*16/(N*K/2):.1f}x  <- Marlin 的 packed 布局有填充")