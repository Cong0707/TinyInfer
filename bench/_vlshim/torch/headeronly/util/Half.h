#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// c10::Half must be a DISTINCT type: vLLM specialises PackedTypeConverter for both half and
// c10::Half, so aliasing them makes the two partial specialisations identical.
struct _vl_c10_half {
  __half v;
  _vl_c10_half() = default;
  _vl_c10_half(__half x) : v(x) {}
  operator __half() const { return v; }
};
struct _vl_c10_bf16 {
  __nv_bfloat16 v;
  _vl_c10_bf16() = default;
  _vl_c10_bf16(__nv_bfloat16 x) : v(x) {}
  operator __nv_bfloat16() const { return v; }
};
namespace c10 { using Half = _vl_c10_half; using BFloat16 = _vl_c10_bf16; }
namespace torch { namespace headeronly { using Half = _vl_c10_half; using BFloat16 = _vl_c10_bf16; } }
