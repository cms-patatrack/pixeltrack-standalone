#ifndef AlpakaCore_initialise_h
#define AlpakaCore_initialise_h

#include <alpaka/alpaka.hpp>

// Initialise the platform and devices for each backend.
// Note: these functions are not thread-safe

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT
namespace alpaka_serial_sync {
  void initialise(bool verbose = false);
}
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_PRESENT
namespace alpaka_tbb_async {
  void initialise(bool verbose = false);
}
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_PRESENT
namespace alpaka_cuda_async {
  void initialise(bool verbose = false);
}
#endif

#ifdef ALPAKA_ACC_GPU_HIP_PRESENT
namespace alpaka_rocm_async {
  void initialise(bool verbose = false);
}
#endif

#endif  // AlpakaCore_initialise_h
