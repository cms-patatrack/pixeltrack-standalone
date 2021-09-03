#ifndef HeterogeneousCore_CUDAUtilities_Context_h
#define HeterogeneousCore_CUDAUtilities_Context_h

#include "CUDACore/allocate_device.h"
#include "CUDACore/allocate_host.h"

namespace cms::cuda {
  class HostAllocatorContext {
  public:
    explicit HostAllocatorContext(cudaStream_t stream) : stream_(stream) {}

    void *allocate_host(size_t nbytes) const { return cms::cuda::allocate_host(nbytes, stream_); }

    void free_host(void *ptr) const { cms::cuda::free_host(ptr); }

  private:
    cudaStream_t stream_;
  };

  class DeviceAllocatorContext {
  public:
    explicit DeviceAllocatorContext(cudaStream_t stream) : stream_(stream) {}

    void *allocate_device(size_t nbytes) const { return cms::cuda::allocate_device(nbytes, stream_); }

    void free_device(void *ptr) const { cms::cuda::free_device(ptr, stream_); }

  private:
    cudaStream_t stream_;
  };

  class Context {
  public:
    explicit Context(cudaStream_t stream) : stream_(stream) {}

    cudaStream_t stream() const { return stream_; }

    operator HostAllocatorContext() const { return HostAllocatorContext(stream()); }
    operator DeviceAllocatorContext() const { return DeviceAllocatorContext(stream()); }

  private:
    cudaStream_t stream_;
  };
}  // namespace cms::cuda

#endif
