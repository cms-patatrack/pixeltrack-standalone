#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SYCLCore/ScopedSetDevice.h"

namespace cms::sycl {
  void *allocate_device(::sycl::queue stream, size_t nbytes) {
    return ::sycl::malloc_device(nbytes, stream);
  }

  void free_device(::sycl::queue stream,  void *ptr) {
    ::sycl::free(ptr, stream);
  }

}  // namespace cms::sycl
