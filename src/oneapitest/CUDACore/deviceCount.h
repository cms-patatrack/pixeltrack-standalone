#ifndef HeterogenousCore_CUDAUtilities_deviceCount_h
#define HeterogenousCore_CUDAUtilities_deviceCount_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CUDACore/cudaCheck.h"

namespace cms {
  namespace cuda {
    inline int deviceCount() {
      int ndevices;
      /*
      DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((ndevices = dpct::dev_mgr::instance().device_count(), 0));
      return ndevices;
    }
  }  // namespace cuda
}  // namespace cms

#endif
