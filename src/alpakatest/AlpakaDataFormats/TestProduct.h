#ifndef AlpakaDataFormats_TestProduct_h
#define AlpakaDataFormats_TestProduct_h

#include <any>
#include <memory>

// Template just to get something weakly Alpaka-dependent in the type
template <typename Queue>
class TestProduct {
public:
  template <typename T>
  TestProduct(std::shared_ptr<Queue> queue, T data)
    : queue_(std::move(queue)),
      data_(std::move(data)) {}

  const Queue& queue() const { return *queue_; }

  template <typename T>
  const T& get() const {
    return std::any_cast<const T&>(data_);
  }

private:
  std::shared_ptr<Queue> queue_;
  std::any data_;
};

#endif
