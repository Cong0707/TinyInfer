
// Benchmark vLLM's Marlin kernel on this GPU at our expert GEMM shape.
#include "kernel.h"
#include "marlin_template.h"

#define LAUNCH_MARLIN(TM, TN, TK, GB, TPB)                                        \
  do {                                                                           \
    marlin::Marlin<                                                              \
        vllm::kFloat16.id(), vllm::kUint4b8.id(),                                \
        vllm::kFloat16.id(), vllm::kFloat16.id(),                                \
        TPB, TM, TN, TK, false, 4, GB, false>                                    \
        <<<blocks, TPB, max_shared_mem, stream>>>(                               \
            A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_scales_ptr, s_ptr,       \
            global_scale_ptr, zp_ptr, prob_m, prob_n, prob_k, lda, locks,        \
            has_bias, use_atomic_add, use_fp32_reduce, max_shared_mem);          \
    return 0;                                                                    \
  } while (0)

extern "C" int marlin_test_launch(
    const void* A, const void* B, void* C, void* C_tmp, const void* bias,
    const float* a_scales, const void* s, const float* global_scale,
    const void* zp, int prob_m, int prob_n, int prob_k, int lda, int* locks,
    bool has_bias, bool use_atomic_add, bool use_fp32_reduce, int max_shared_mem,
    cudaStream_t stream, int blocks) {
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* bias_ptr = (const int4*)bias;
  const int4* s_ptr = (const int4*)s;
  const int4* zp_ptr = (const int4*)zp;
  const float* a_scales_ptr = a_scales;
  const float* global_scale_ptr = global_scale;

  LAUNCH_MARLIN(1, 8, 8, -1, 128);
  LAUNCH_MARLIN(1, 16, 4, -1, 128);
  return 1;
}