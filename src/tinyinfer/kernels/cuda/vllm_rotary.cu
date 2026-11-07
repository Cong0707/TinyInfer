// 从 vLLM csrc/pos_encoding_kernels.cu 移植（rotary_embedding，IS_NEOX=false 分支）。
//
// 上游结构保持不变：
//   apply_token_rotary_embedding  —— 一个 (x, y) 对的旋转，交错布局
//   apply_rotary_embedding        —— 一个 token 的所有 head
//   rotary_embedding_kernel       —— 一个 block 一个 token，位置从 positions[] 取
//
// 与上游的差异只有落地方式：这里不带 torch 头文件（我们的 native.py 直接加载
// cubin），所以模板参数固化成 bf16 元素 + float 的 cos/sin 表，指针换成裸指针。
//
// 上游用 cos_sin_cache[pos * rot_dim] 取表，即表按 position 索引、每个 position
// 存 [cos(rot_dim/2), sin(rot_dim/2)]。这一条比我们之前自己写的那版严格更好：
// 那版要求表只有一行（s == 1），prefill 根本用不了。
//
// 布局：交错（GPT-J 风格），x_index = 2*rot_offset，y_index = 2*rot_offset+1。
#include <cuda_bf16.h>

__device__ __forceinline__ void token_rope_bf(
    __nv_bfloat16* __restrict__ arr, const float* __restrict__ cos_ptr,
    const float* __restrict__ sin_ptr, int rot_offset, int embed_dim,
    const bool inverse) {
  const int x_index = 2 * rot_offset;
  const int y_index = 2 * rot_offset + 1;
  float cos_f = __ldg(cos_ptr + rot_offset);
  float sin_f = __ldg(sin_ptr + rot_offset);
  if (inverse) {
    sin_f = -sin_f;
  }
  const float x_f = __bfloat162float(arr[x_index]);
  const float y_f = __bfloat162float(arr[y_index]);
  arr[x_index] = __float2bfloat16(x_f * cos_f - y_f * sin_f);
  arr[y_index] = __float2bfloat16(y_f * cos_f + x_f * sin_f);
}

extern "C" __global__ void vllm_rotary_embedding_bf(
    const long long* __restrict__ positions,   // [num_tokens]
    __nv_bfloat16* __restrict__ query,         // [num_tokens, num_heads, head_size]
    const float* __restrict__ cos_sin_cache,   // [num_position, rot_dim]
    int rot_dim, long long query_stride,
    long long head_stride, int num_heads, int head_size,
    long long rope_dim_offset, int inverse) {
  const int token_idx = blockIdx.x;
  const long long pos = positions[token_idx];
  const float* cache_ptr = cos_sin_cache + pos * rot_dim;
  const int embed_dim = rot_dim / 2;
  const float* cos_ptr = cache_ptr;
  const float* sin_ptr = cache_ptr + embed_dim;
  const bool inv = inverse != 0;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const long long token_head =
        (long long)token_idx * query_stride + (long long)head_idx * head_stride + rope_dim_offset;
    const int rot_offset = i % embed_dim;
    token_rope_bf(query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim, inv);
  }

}
