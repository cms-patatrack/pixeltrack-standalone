#ifndef HeterogenousCore_CUDAUtilities_currentDevice_h
#define HeterogenousCore_CUDAUtilities_currentDevice_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace cms {
  namespace cuda {
    inline int currentDevice() {
      return dpct::dev_mgr::instance().current_device_id();
    }
  }  // namespace cuda
}  // namespace cms

#endif
