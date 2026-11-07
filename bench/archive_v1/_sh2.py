print("=== Marlin shared 精确需求 ===")
print()
# thread_m=1, thread_n=16, thread_k=4, stages=4, threads=128
thread_n_blocks = 16
thread_k_blocks = 4
pack_factor = 8
m_block_size = 16
is_a_8bit = 0
stages = 4

a_sh_stride = 16 * thread_k_blocks // (16 if is_a_8bit else 8)
a_sh_stage = a_sh_stride * m_block_size

b_sh_stride = ((thread_n_blocks * 16) * 16 // pack_factor) // (2 if is_a_8bit else 4)
b_sh_stage = b_sh_stride * thread_k_blocks // (2 if is_a_8bit else 1)

print(f"a_sh_stride = {a_sh_stride}, a_sh_stage = {a_sh_stage}")
print(f"b_sh_stride = {b_sh_stride}, b_sh_stage = {b_sh_stage}")
print()
# shared layout: sh_a[stages*a_sh_stage] then sh_b[stages*b_sh_stage], then scales etc
a_int4 = stages * a_sh_stage
b_int4 = stages * b_sh_stage
print(f"sh_a: {stages} x {a_sh_stage} = {a_int4} int4 = {a_int4*16} bytes")
print(f"sh_b: {stages} x {b_sh_stage} = {b_int4} int4 = {b_int4*16} bytes")
print(f"subtotal = {(a_int4+b_int4)*16} bytes")
print()
print("实测越界到 0x9400 = 37888 -> 需要 >= 37888")
print(f"  计算值 {(a_int4+b_int4)*16} 偏小，说明还有 scale/zp 的 shared 段")
print()
print("直接取 37888 并留余量到 40960")