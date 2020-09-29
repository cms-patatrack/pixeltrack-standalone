#ifndef HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h
#define HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CUDACore/cudaCheck.h"

namespace cms {
  namespace cuda {
    class ScopedSetDevice {
    public:
      explicit ScopedSetDevice(int newDevice) {
        cudaCheck(prevDevice_ = dpct::dev_mgr::instance().current_device_id());
        /*
        DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
        */
        cudaCheck((dpct::dev_mgr::instance().select_device(newDevice), 0));
      }

      ~ScopedSetDevice() {
        // Intentionally don't check the return value to avoid
        // exceptions to be thrown. If this call fails, the process is
        // doomed anyway.
        dpct::dev_mgr::instance().select_device(prevDevice_);
      }

    private:
      int prevDevice_;
    };
  }  // namespace cuda
}  // namespace cms

#endif
