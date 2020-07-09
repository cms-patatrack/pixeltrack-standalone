#ifndef HeterogeneousCore_CUDAUtilities_allocate_host_h
#define HeterogeneousCore_CUDAUtilities_allocate_host_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace cms {
  namespace cuda {
    // Allocate pinned host memory (to be called from unique_ptr)
    void *allocate_host(size_t nbytes, sycl::queue *stream);

    // Free pinned host memory (to be called from unique_ptr)
    void free_host(void *ptr);
  }  // namespace cuda
}  // namespace cms

#endif
