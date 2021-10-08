#ifndef HeterogeneousCore_CUDACore_ESContext_h
#define HeterogeneousCore_CUDACore_ESContext_h

#include "CUDACore/Context.h"
#include "CUDACore/ESProductNew.h"
#include "CUDACore/SharedStreamPtr.h"
#include "CUDACore/StreamCache.h"

namespace cms::cuda {
  class ESContext {
  public:
    explicit ESContext(int device);

    int device() const { return currentDevice_; }

    cudaStream_t stream() const { return stream_.get(); }

    operator HostAllocatorContext() { return HostAllocatorContext(stream()); }
    operator DeviceAllocatorContext() { return DeviceAllocatorContext(stream()); }
    operator Context() { return Context(stream()); }

  private:
    int currentDevice_;
    SharedStreamPtr stream_;
  };

  namespace impl {
    template <typename T>
    class RunForEachDevice {
    public:
      RunForEachDevice(T&& data) : data_(std::forward<T>(data)) {}

      template <typename F>
      [[nodiscard]] auto forEachDevice(F const& func) {
        using RetType = decltype(func(std::declval<const T&>(), std::declval<ESContext&>()));
        static_assert(not std::is_same_v<RetType, void>,
                      "Function must return a value corresponding the ESProduct for one device");
        auto product = std::make_unique<cms::cudaNew::ESProduct<RetType>>();
        auto const& cref = data_;
        for (std::size_t i = 0; i < product->size(); ++i) {
          ESContext ctx(i);
          product->emplace(i, func(cref, ctx), ctx.stream());
        }
        product->setHostData(std::move(data_));
        return product;
      }

    private:
      T data_;
    };
  }  // namespace impl

  template <typename F>
  [[nodiscard]] auto runForHost(F&& func) {
    using RetType = decltype(func(std::declval<HostAllocatorContext&>()));
    static_assert(not std::is_same_v<RetType, void>,
                  "Function must return an intermediate value passed to the functor argument of runOnEachDevice()");
    // TODO: temporarily use a "random stream" for the pinned host
    // memory allocation until we figure out a way to use the caching
    // allocator without requiring a stream. The
    // cms::cudaNew::ESProduct will take care of releasing any
    // allocated host memory only after all asynchronous work issued
    // in RunForEachDevice::forEachDevice() has completed.
    cudaCheck(cudaSetDevice(0));
    auto stream = getStreamCache().get();
    HostAllocatorContext ctx(stream.get());
    return impl::RunForEachDevice<RetType>(func(ctx));
  }

  template <typename F>
  [[nodiscard]] auto runForEachDevice(F&& func) {
    using RetType = decltype(func(std::declval<ESContext&>()));
    static_assert(not std::is_same_v<RetType, void>,
                  "Function must return a value corresponding the ESProduct for noe device");
    auto product = std::make_unique<cms::cudaNew::ESProduct<RetType>>();
    for (int i = 0; i < product.size(); ++i) {
      ESContext ctx(i);
      product->emplace(i, func(ctx), ctx.stream());
    }
    return product;
  }

}  // namespace cms::cuda

#endif
