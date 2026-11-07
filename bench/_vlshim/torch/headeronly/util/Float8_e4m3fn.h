#pragma once
#include <cuda_fp8.h>
// vLLM calls T::from_bits(); CUDA's __nv_fp8_e4m3 has no such method, so wrap it.
struct _vl_fp8_e4m3 {
  __nv_fp8_e4m3 v;
  _vl_fp8_e4m3() = default;
  _vl_fp8_e4m3(__nv_fp8_e4m3 x) : v(x) {}
  operator __nv_fp8_e4m3() const { return v; }
  static _vl_fp8_e4m3 from_bits(unsigned short b) {
    _vl_fp8_e4m3 r; __nv_fp8_storage_t s = (__nv_fp8_storage_t)b;
    r.v = *reinterpret_cast<__nv_fp8_e4m3*>(&s); return r;
  }
  unsigned short bitcast() const { return *reinterpret_cast<const __nv_fp8_storage_t*>(&v); }
};
namespace c10 {
using Float8_e4m3fn = _vl_fp8_e4m3;
using Float8_e4m3fnuz = _vl_fp8_e4m3;
using Float8_e5m2 = _vl_fp8_e4m3;
}
namespace torch { namespace headeronly {
using Float8_e4m3fn = _vl_fp8_e4m3;
using Float8_e4m3fnuz = _vl_fp8_e4m3;
} }
