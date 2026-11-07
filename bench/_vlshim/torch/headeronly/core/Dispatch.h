#pragma once
// torch 2.9 does not ship torch/headeronly/core/Dispatch.h (vLLM 0.33 is built against a
// newer torch). The macros vLLM's dispatch_utils.h needs are the THO_* family, which for
// this purpose are the ATen dispatch macros applied to torch::headeronly::ScalarType.
// Re-derive them here from ATen's Dispatch.h so the vLLM kernels compile unmodified.
#include <ATen/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>

#define THO_DISPATCH_SWITCH(TYPE, NAME, ...) AT_DISPATCH_SWITCH(TYPE, NAME, __VA_ARGS__)
#define THO_DISPATCH_CASE(...) AT_DISPATCH_CASE(__VA_ARGS__)
#define THO_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, __VA_ARGS__)
#define THO_DISPATCH_CASE_INTEGRAL_TYPES(...) AT_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__)
#define THO_DISPATCH_CASE_FLOATING_TYPES(...) AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)
#define THO_DISPATCH_CASE_INTEGRAL_AND_UNSIGNED_TYPES(...) \
  AT_DISPATCH_CASE_INTEGRAL_AND_UNSIGNED_TYPES(__VA_ARGS__)
