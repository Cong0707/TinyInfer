#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "torch/headeronly/core/ScalarType.h"
#include "tensor_stub.h"

#define VLLM_STABLE_DISPATCH_RANK234(RANK, F) \
  switch (RANK) { \
    case 2: { constexpr int tensor_rank = 2; F(); break; } \
    case 3: { constexpr int tensor_rank = 3; F(); break; } \
    case 4: { constexpr int tensor_rank = 4; F(); break; } \
    default: break; \
  }

#define VLLM_STABLE_DISPATCH_FLOATING_TYPES(TYPE, NAME, F) \
  switch (TYPE) { \
    case torch::headeronly::ScalarType::Half: { using scalar_t = _vl_c10_half; F(); break; } \
    case torch::headeronly::ScalarType::BFloat16: { using scalar_t = _vl_c10_bf16; F(); break; } \
    case torch::headeronly::ScalarType::Float: { using scalar_t = float; F(); break; } \
    default: break; \
  }

#define VLLM_STABLE_DISPATCH_VEC_SIZE(VS, F) \
  switch (VS) { \
    case 1: { constexpr int vec_size = 1; F(); break; } \
    case 2: { constexpr int vec_size = 2; F(); break; } \
    case 4: { constexpr int vec_size = 4; F(); break; } \
    case 8: { constexpr int vec_size = 8; F(); break; } \
    default: break; \
  }
