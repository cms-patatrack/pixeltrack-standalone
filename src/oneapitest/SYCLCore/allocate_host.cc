#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace cms::sycl {
  void *allocate_host(size_t nbytes, ::sycl::queue stream) {
    return ::sycl::malloc_host(nbytes, stream);
  }

  void free_host(void *ptr) {
    ::sycl::free(ptr, dpct::get_default_queue());
  }

}  // namespace cms::sycl
