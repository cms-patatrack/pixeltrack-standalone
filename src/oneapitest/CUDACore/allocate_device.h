#ifndef HeterogeneousCore_CUDAUtilities_allocate_device_h
#define HeterogeneousCore_CUDAUtilities_allocate_device_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace cms {
  namespace cuda {
    // Allocate device memory
    void *allocate_device(int dev, size_t nbytes, sycl::queue *stream);

    // Free device memory (to be called from unique_ptr)
    void free_device(int device, void *ptr);
  }  // namespace cuda
}  // namespace cms

#endif
