#ifndef HeterogenousCore_SYCLUtilities_currentDevice_h
#define HeterogenousCore_SYCLUtilities_currentDevice_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace cms {
  namespace sycl {
    inline int currentDevice() {
      return dpct::dev_mgr::instance().current_device_id();
    }
  }  // namespace sycl
}  // namespace cms

#endif
