#pragma once
#include <cstdint>
#include <vector>
#include <optional>
#include <cuda_runtime.h>
#include "torch/headeronly/core/ScalarType.h"

namespace torch { namespace stable {

enum class DeviceType : int { CPU = 0, CUDA = 1 };
struct Device {
  int index = 0; DeviceType type = DeviceType::CUDA;
  Device() = default; Device(int i) : index(i) {}
  bool is_cuda() const { return true; }
  bool is_cpu() const { return false; }
  int get_device_index() const { return index; }
};

// reuse the headeronly enum so dispatch macros match
using ScalarType = torch::headeronly::ScalarType;

struct Tensor {
  void* ptr = nullptr;
  int64_t sizes_[4] = {0,0,0,0};
  int64_t strides_[4] = {1,1,1,1};
  int dim_ = 0;
  ScalarType st_ = ScalarType::Float;
  Tensor() = default;
  Tensor(void* p, std::vector<int64_t> sz, ScalarType st = ScalarType::Float) : ptr(p), st_(st) {
    dim_ = (int)sz.size();
    int64_t s = 1;
    for (int i = dim_ - 1; i >= 0; --i) { sizes_[i] = sz[i]; strides_[i] = s; s *= sz[i]; }
  }
  int dim() const { return dim_; }
  int64_t size(int i) const { if (i < 0) i += dim_; return (i >= 0 && i < 4) ? sizes_[i] : 0; }
  int64_t stride(int i) const { if (i < 0) i += dim_; return (i >= 0 && i < 4) ? strides_[i] : 1; }
  int64_t numel() const { int64_t n = 1; for (int i = 0; i < dim_; ++i) n *= sizes_[i]; return n; }
  template <typename T = void> T* data_ptr() const { return static_cast<T*>(ptr); }
  template <typename T = void> const T* const_data_ptr() const { return static_cast<const T*>(ptr); }
  template <typename T = void> T* mutable_data_ptr() const { return static_cast<T*>(ptr); }
  int get_device_index() const { return 0; }
  Device device() const { return Device(0); }
  bool is_contiguous() const { return true; }
  ScalarType scalar_type() const { return st_; }
};

namespace accelerator {
struct DeviceGuard { explicit DeviceGuard(int) {} };
struct DeviceGuardGuard { explicit DeviceGuardGuard(int) {} };
}

inline Tensor contiguous(const Tensor& t) { return t; }
inline Tensor to(const Tensor& t, ScalarType) { return t; }
inline Tensor clone(const Tensor& t) { return t; }
inline cudaStream_t get_current_cuda_stream() { return 0; }
} }

inline cudaStream_t get_current_cuda_stream() { return 0; }
inline int get_device_prop(int) { return 0; }
#define STD_TORCH_CHECK(...) ((void)0)
#define STD_TORCH_CHECK_NOT_IMPLEMENTED(...) ((void)0)
