#pragma once
#include <stdexcept>
#include <string>
#include <sstream>
namespace torch { namespace headeronly {
class Error : public std::exception {
 public:
  explicit Error(std::string m) : msg_(std::move(m)) {}
  const char* what() const noexcept override { return msg_.c_str(); }
 private:
  std::string msg_;
};
} }
#define STD_TORCH_CHECK(cond, ...) ((void)0)
#define STD_TORCH_CHECK_NOT_IMPLEMENTED(...) ((void)0)
