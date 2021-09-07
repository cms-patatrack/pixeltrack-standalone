#ifndef HeterogeneousCore_CUDAUtilities_TestContext_h
#define HeterogeneousCore_CUDAUtilities_TestContext_h

#include "CUDACore/Context.h"
#include "CUDACore/currentDevice.h"

namespace cms::cudatest {
  class TestContext {
  public:
    TestContext() : TestContext(cudaStreamDefault) {}
    explicit TestContext(cudaStream_t stream) : stream_{stream} {}

    operator cms::cuda::HostAllocatorContext() const { return cms::cuda::HostAllocatorContext(stream_); }
    operator cms::cuda::DeviceAllocatorContext() const { return cms::cuda::DeviceAllocatorContext(stream_); }
    operator cms::cuda::Context() const { return cms::cuda::Context(stream_); }

  private:
    cudaStream_t stream_;
  };
}  // namespace cms::cudatest

#endif
