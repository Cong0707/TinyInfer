#pragma once
#include <cstdint>
namespace torch { namespace headeronly {
enum class ScalarType : int8_t { Byte=0, Char=1, Short=2, Int=3, Long=4, Half=5, Float=6, Double=7, Bool=11, Float8_e4m3fn=12, Float8_e5m2=13, BFloat16=15 };
} }
namespace c10 { using ScalarType = torch::headeronly::ScalarType; }
