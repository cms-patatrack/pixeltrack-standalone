#ifndef HeterogeneousCore_CUDAUtilities_allocate_managed_h
#define HeterogeneousCore_CUDAUtilities_allocate_managed_h

#include <cuda_runtime.h>

namespace cms {
  namespace cuda {
    // Allocate managed memory (to be called from unique_ptr)
    void *allocate_managed(size_t nbytes, cudaStream_t stream);

    // Free managed memory (to be called from unique_ptr)
    void free_managed(void *ptr);
  }  // namespace cuda
}  // namespace cms

#endif
