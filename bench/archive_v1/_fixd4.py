from pathlib import Path
p = Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\native\gemv_fp4_e8m0.cu")
src = p.read_text(encoding="utf-8")
# insert d4 before the fast4 kernel (it exists elsewhere in the file but later)
d4 = r'''
// E2M1 code -> float with no memory access: e>0 is 2^(e-1)*(1+m/2), the float whose
// exponent field is 126+e and whose mantissa top bit is m. e==0 is the subnormal 0 or 0.5.
__device__ __forceinline__ float d4f(unsigned int c) {
    const unsigned int e = (c >> 1) & 3u;
    const unsigned int m = c & 1u;
    const unsigned int se = e | (e == 0u);
    float v = __uint_as_float(((126u + se) << 23) | (m << 22));
    v = (e == 0u) ? (m ? 0.5f : 0.f) : v;
    return (c & 8u) ? -v : v;
}
'''
i = src.find('extern "C" __global__ void gemv_fp4_fast4')
src = src[:i] + d4 + "\n" + src[i:]
src = src.replace("d4(word", "d4f(word").replace("d4((word", "d4f((word")
p.write_text(src, encoding="utf-8")
print("ok")
