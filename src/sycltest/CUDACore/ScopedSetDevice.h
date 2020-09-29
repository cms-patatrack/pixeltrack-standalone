#ifndef HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h
#define HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace cms {
  namespace cuda {
    class ScopedSetDevice {
    public:
      explicit ScopedSetDevice(int newDevice) {
        prevDevice_ = dpct::dev_mgr::instance().current_device_id();
        dpct::dev_mgr::instance().select_device(newDevice);
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
