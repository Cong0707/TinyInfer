#include "dispatch_utils.h"

#include "core/utils.hpp"
#include "core/cuda_compat.h"

#include "quantization/vectorization_utils.cuh"
#include "concat_mla_q.cuh"

#if !defined(USE_ROCM) && defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
  #include "nvfp4_ds_mla_cache.h"
#endif

#ifdef USE_ROCM
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <cstdlib>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#else
  #include <cuda.h>
#endif

#if defined(__gfx942__)
constexpr float kFp8ScaleDivisor = 224.f;
#else
constexpr float kFp8ScaleDivisor = 448.f;
#endif

__global__ void _vl_anchor() {}
