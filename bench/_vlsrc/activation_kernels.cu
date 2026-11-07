#include <cuda.h>
#include <cuda_runtime.h>

#ifndef USE_ROCM
  #include <cuda_fp8.h>
#endif

#include <cmath>
#include <type_traits>

#include "core/cuda_compat.h"
#include "async_util.cuh"
#include "cuda_vec_utils.cuh"
#include "dispatch_utils.h"

namespace vllm {

// `alpha` and `beta` are applied to opposite operands:
//   - alpha lives INSIDE the activation (the activated half): the gated
//     activation computes act_half * sigmoid(alpha * act_half).
//   - beta is added to the OTHER (non-activated) half before the multiply.
// So the result is always ACT(act_half, alpha) * (other_half + beta).
// Which half is which depends on `act_first` (see below). Defaults
// alpha=1.0, beta=0.0 reproduce the plain SwiGLU/GeGLU behavior.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&, const float),
          bool act_first, bool HAS_CLAMP>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y,
                                            const float limit,
                                            const float alpha,
                                            const float beta) {
  if constexpr (act_first) {
    scalar_t gate = x;
    scalar_t up = y;
    if constexpr (HAS_CLAMP) {
      gate = (scalar_t)fminf((float)gate, limit);
      up = (scalar_t)fmaxf(fminf((float)up, limit), -limit);
    }
    // act_first: gate is the activated half -> alpha applies to gate;
    // beta is added to up (the non-activated half).
    return (scalar_t)(ACT_FN(gate, alpha) * ((float)up + beta));
  } else {
    scalar_t gate = x;
    scalar_t up = y;
    if constexpr (HAS_CLAMP) {
      gate = (scalar_t)fmaxf(fminf((float)gate, limit), -limit);
      up = (scalar_t)fminf((float)up, limit);
    }
    // !act_first: up is the activated half -> alpha applies to up;
    // beta is added to gate (the non-activated half).
    return (scalar_t)(((float)gate + beta) * ACT_FN(up, alpha));
  }
}

template <typename packed_t,
          packed_t (*PACKED_ACT_FN)(const packed_t&, const float),
          bool act_first, bool HAS_CLAMP>
__device__ __forceinline__ packed_t packed_compute(const packed_t& x,
                                                   const packed_t& y,
                                                   const float limit,
                                                   const float alpha,
                                                   const float beta) {
  if constexpr (act_first) {
    packed_t gate = x;
    packed_t up = y;
    float2 u = cast_to_float2(up);
    if constexpr (HAS_CLAMP) {
      float2 g = cast_to_float2(gate);
      g.x = fminf(g.x, limit);
      g.y = fminf(g.y, limit);
      u.x = fmaxf(fminf(u.x, limit), -limit);
      u.y = fmaxf(fminf(u.y, limit), -limit);
      gate = cast_to_packed<packed_t>(g);
    }
    // act_first: gate is the activated half -> alpha applies to gate;
    // beta is added to up (the non-activated half).
    float2 activated = cast_to_float2(PACKED_ACT_FN(gate, alpha));
    activated.x *= u.x + beta;
    activated.y *= u.y + beta;
    return cast_to_packed<packed_t>(activated);
  } else {
    packed_t gate = x;
    packed_t up = y;
    float2 g = cast_to_float2(gate);
    if constexpr (HAS_CLAMP) {
      float2 u = cast_to_float2(up);
      g.x = fmaxf(fminf(g.x, limit), -limit);
      g.y = fmaxf(fminf(g.y, limit), -limit);
      u.x = fminf(u.x, limit);
      u.y = fminf(u.y, limit);
      up = cast_to_packed<packed_t>(u);
    }
    // !act_first: up is the activated half -> alpha applies to up;
    // beta is added to gate (the non-activated half).
    float2 activated = cast_to_float2(PACKED_ACT_FN(up, alpha));
    activated.x *= g.x + beta;
    activated.y *= g.y + beta;
    return cast_to_packed<packed_t>(activated);
  }
}

// Activation and gating kernel template.
template <typename scalar_t, typename packed_t,
          scalar_t (*ACT_FN)(const scalar_t&, const float),
          packed_t (*PACKED_ACT_FN)(const packed_t&, const float),
          bool act_first, bool use_vec, bool HAS_CLAMP, bool use_256b = false>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d, const float limit, const float alpha, const float beta) {
  const int64_t token_idx = blockIdx.x;
  const scalar_t* x_ptr = input + token_idx * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + token_idx * d;

  if constexpr (use_vec) {
    using cuda_t = typename CUDATypeConverter<scalar_t>::Type;
    using pvec_t = PackedVec<cuda_t, use_256b>;

    const pvec_t* x_vec = reinterpret_cast<const pvec_t*>(x_ptr);
    const pvec_t* y_vec = reinterpret_cast<const pvec_t*>(y_ptr);
    pvec_t* out_vec = reinterpret_cast<pvec_t*>(out_ptr);
    const int num_vecs = d / 2 / pvec_t::NUM_ELTS;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      pvec_t x, y;
      if constexpr (use_256b) {
        ld256(x, &x_vec[i]);
        ld256(y, &y_vec[i]);
      } else {
        ld128(x, &x_vec[i]);
        ld128(y, &y_vec[i]);
      }
#pragma unroll
      for (int j = 0; j < pvec_t::NUM_ELTS; j++) {
        x.elts[j] =
            packed_compute<packed_t, PACKED_ACT_FN, act_first, HAS_CLAMP>(
                x.elts[j], y.elts[j], limit, alpha, beta);
      }
      if constexpr (use_256b) {
        st256(x, &out_vec[i]);
      } else {
        st128(x, &out_vec[i]);
      }
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&x_ptr[idx]);
      const scalar_t y = VLLM_LDG(&y_ptr[idx]);
      out_ptr[idx] = compute<scalar_t, ACT_FN, act_first, HAS_CLAMP>(
          x, y, limit, alpha, beta);
    }
  }
}

// Gated activations take an `alpha` argument that scales the sigmoid input
// (`x * sigmoid(alpha * x)`). alpha defaults to 1.0 at all call sites, which
// is exactly SiLU; only the clamp path (silu_and_mul_with_clamp) passes a
// non-default alpha. Activations that do not use alpha simply ignore it.
template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x, const float alpha) {
  // x * sigmoid(alpha * x)
  return (T)(((float)x) / (1.0f + expf((float)-x * alpha)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_silu_kernel(const packed_t& val,
                                                       const float alpha) {
  // x * sigmoid(alpha * x)
  float2 fval = cast_to_float2(val);
  fval.x = fval.x / (1.0f + expf(-fval.x * alpha));
  fval.y = fval.y / (1.0f + expf(-fval.y * alpha));
  return cast_to_packed<packed_t>(fval);
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x, const float /*alpha*/) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_gelu_kernel(const packed_t& val,
                                                       const float /*alpha*/) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  constexpr float ALPHA = M_SQRT1_2;
  float2 fval = cast_to_float2(val);
  fval.x = fval.x * 0.5f * (1.0f + ::erf(fval.x * ALPHA));
  fval.y = fval.y * 0.5f * (1.0f + ::erf(fval.y * ALPHA));
  return cast_to_packed<packed_t>(fval);
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x,
                                              const float /*alpha*/) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t
packed_gelu_tanh_kernel(const packed_t& val, const float /*alpha*/) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  float2 fval = cast_to_float2(val);
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;

  float x_cube = fval.x * fval.x * fval.x;
  float inner = BETA * (fval.x + KAPPA * x_cube);
  fval.x = 0.5f * fval.x * (1.0f + ::tanhf(inner));

  x_cube = fval.y * fval.y * fval.y;
  inner = BETA * (fval.y + KAPPA * x_cube);
  fval.y = 0.5f * fval.y * (1.0f + ::tanhf(inner));
  return cast_to_packed<packed_t>(fval);
}

}  // namespace vllm

// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first. HAS_CLAMP (bool) enables pre-activation clamping: gate input is
// clamped (max only) and up input is clamped (both sides) before the
// activation function is applied.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, PACKED_KERNEL, ACT_FIRST,        \
                                      HAS_CLAMP, LIMIT, ALPHA, BETA)           \
  auto dtype = input.scalar_type();                                            \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  dim3 grid(num_tokens);                                                       \
  int cc_major = get_device_prop()->major;                                     \
  int support_vec =                                                            \
      (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)            \
          ? vllm::VecTraits<true>::ARCH_MAX_VEC_SIZE                           \
          : vllm::VecTraits<false>::ARCH_MAX_VEC_SIZE;                         \
  int vec_size = support_vec / input.element_size();                           \
  const bool use_vec = (d % vec_size == 0);                                    \
  const torch::stable::accelerator::DeviceGuard device_guard(                  \
      input.get_device_index());                                               \
  const cudaStream_t stream = get_current_cuda_stream();                       \
  if (use_vec) {                                                               \
    dim3 block(std::min(d / vec_size, 1024));                                  \
    if (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128) {         \
      VLLM_STABLE_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {   \
        vllm::act_and_mul_kernel<                                              \
            scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,      \
            KERNEL<scalar_t>,                                                  \
            PACKED_KERNEL<typename vllm::PackedTypeConverter<scalar_t>::Type>, \
            ACT_FIRST, true, HAS_CLAMP, true><<<grid, block, 0, stream>>>(     \
            out.mutable_data_ptr<scalar_t>(),                                  \
            input.const_data_ptr<scalar_t>(), d, LIMIT, ALPHA, BETA);          \
      });                                                                      \
    } else {                                                                   \
      VLLM_STABLE_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {   \
        vllm::act_and_mul_kernel<                                              \
            scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,      \
            KERNEL<scalar_t>,                                                  \
            PACKED_KERNEL<typename vllm::PackedTypeConverter<scalar_t>::Type>, \
            ACT_FIRST, true, HAS_CLAMP, false><<<grid, block, 0, stream>>>(    \
            out.mutable_data_ptr<scalar_t>(),                                  \
            input.const_data_ptr<scalar_t>(), d, LIMIT, ALPHA, BETA);          \
      });                                                                      \
    }                                                                          \
  } else {                                                                     \
    dim3 block(std::min(d, 1024));                                             \
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {     \
      vllm::act_and_mul_kernel<                                                \
          scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,        \
          KERNEL<scalar_t>,                                                    \
          PACKED_KERNEL<typename vllm::PackedTypeConverter<scalar_t>::Type>,   \
          ACT_FIRST, false, HAS_CLAMP><<<grid, block, 0, stream>>>(            \
          out.mutable_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(),  \
          d, LIMIT, ALPHA, BETA);                                              \
    });                                                                        \
  }

__global__ void _vl_anchor() {}
