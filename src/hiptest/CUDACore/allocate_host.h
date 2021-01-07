#ifndef HeterogeneousCore_CUDAUtilities_allocate_host_h
#define HeterogeneousCore_CUDAUtilities_allocate_host_h

#include <hip/hip_runtime.h>

namespace cms {
  namespace cuda {
    // Allocate pinned host memory (to be called from unique_ptr)
    void *allocate_host(size_t nbytes, hipStream_t stream);

    // Free pinned host memory (to be called from unique_ptr)
    void free_host(void *ptr);
  }  // namespace cuda
}  // namespace cms

#endif
