#ifndef HeterogeneousCore_AlpakaUtilities_ScopedSetDevice_h
#define HeterogeneousCore_AlpakaUtilities_ScopedSetDevice_h

#include <cuda_runtime.h>

namespace cms {
  namespace alpakatools {
    class ScopedSetDevice {
    public:
      explicit ScopedSetDevice(int newDevice) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        cudaGetDevice(&prevDevice_);
        cudaSetDevice(newDevice);
#endif
      }

      ~ScopedSetDevice() {
        // Intentionally don't check the return value to avoid
        // exceptions to be thrown. If this call fails, the process is
        // doomed anyway.
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        cudaSetDevice(prevDevice_);
#endif
      }

    private:
      int prevDevice_;
    };
  }  // namespace alpakatools
}  // namespace cms

#endif
