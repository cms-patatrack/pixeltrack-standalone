#ifndef HeterogenousCore_SYCLUtilities_deviceCount_h
#define HeterogenousCore_SYCLUtilities_deviceCount_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace cms {
  namespace sycl {
    inline int deviceCount() {
      return dpct::dev_mgr::instance().device_count();
    }
  }  // namespace sycl
}  // namespace cms

#endif
