#ifndef HeterogenousCore_CUDAUtilities_deviceCount_h
#define HeterogenousCore_CUDAUtilities_deviceCount_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace cms {
  namespace cuda {
    inline int deviceCount() {
      return dpct::dev_mgr::instance().device_count();
    }
  }  // namespace cuda
}  // namespace cms

#endif
