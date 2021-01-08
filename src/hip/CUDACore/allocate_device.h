#ifndef HeterogeneousCore_CUDAUtilities_allocate_device_h
#define HeterogeneousCore_CUDAUtilities_allocate_device_h

#include <hip/hip_runtime.h>

namespace cms {
  namespace hip {
    // Allocate device memory
    void *allocate_device(int dev, size_t nbytes, hipStream_t stream);

    // Free device memory (to be called from unique_ptr)
    void free_device(int device, void *ptr);
  }  // namespace hip
}  // namespace cms

#endif
