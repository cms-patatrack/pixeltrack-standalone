#ifndef HeterogeneousCore_SYCLUtilities_allocate_device_h
#define HeterogeneousCore_SYCLUtilities_allocate_device_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace cms {
  namespace sycl {
    // Allocate device memory
    void *allocate_device(::sycl::queue stream, size_t nbytes);

    // Free device memory (to be called from unique_ptr)
    void free_device(::sycl::queue stream, void *ptr);

  }  // namespace sycl
}  // namespace cms

#endif
