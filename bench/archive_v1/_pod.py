"""Instantiate vLLM's template kernels by keeping their host launchers and swapping types.

Cutting the file at the first torch reference left only templates that nothing instantiated,
so the cubin held nothing but an anchor. The instantiation happens inside the host function
(rms_norm_kernel<...><<<...>>>), so that body must survive.

Approach: keep the whole file, replace torch::stable::Tensor with a tiny POD that has the
handful of methods the launchers call (size/stride/numel/dim/data_ptr/get_device_index), and
provide free functions for the torch::stable helpers. The kernels themselves are untouched.
"""
from pathlib import Path

SHIM = Path(r"E:\Develop\Deepseek\deepseek\_vlshim")
POD = r'''
// Minimal stand-in for torch::stable::Tensor: only what vLLM's launchers call.
#pragma once
#include <cstdint>
#include <vector>
#include <optional>
#include <cuda_runtime.h>

namespace torch { namespace stable {

struct Tensor {
  void* ptr = nullptr;
  int64_t sizes_[4] = {0,0,0,0};
  int64_t strides_[4] = {0,0,0,0};
  int dim_ = 0;

  Tensor() = default;
  Tensor(void* p, std::vector<int64_t> sz) : ptr(p) {
    dim_ = (int)sz.size();
    int64_t st = 1;
    for (int i = dim_ - 1; i >= 0; --i) {
      sizes_[i] = sz[i];
      strides_[i] = st;
      st *= sz[i];
    }
  }
  int dim() const { return dim_; }
  int64_t size(int i) const {
    if (i < 0) i += dim_;
    return (i >= 0 && i < 4) ? sizes_[i] : 0;
  }
  int64_t stride(int i) const {
    if (i < 0) i += dim_;
    return (i >= 0 && i < 4) ? strides_[i] : 0;
  }
  int64_t numel() const {
    int64_t n = 1;
    for (int i = 0; i < dim_; ++i) n *= sizes_[i];
    return n;
  }
  template <typename T = void> T* data_ptr() const { return static_cast<T*>(ptr); }
  int get_device_index() const { return 0; }
  bool is_contiguous() const { return true; }
  Tensor& operator=(const void* p) { ptr = (void*)p; return *this; }
};

namespace accelerator {
struct DeviceGuard { explicit DeviceGuard(int) {} };
}

inline Tensor contiguous(const Tensor& t) { return t; }
inline cudaStream_t get_current_cuda_stream() { return 0; }
} }
'''
(SHIM / "tensor_stub.h").write_text(POD, encoding="utf-8")
print("wrote tensor_stub.h")
