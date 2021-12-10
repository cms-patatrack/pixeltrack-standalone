#include "AlpakaCore/initialise.h"

// these are defined in AlpakaCore/alpaka/initialise.cc

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SUPPORTED
namespace alpaka_serial_sync {
  void initialise();
}
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SUPPORTED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_SUPPORTED
namespace alpaka_tbb_async {
  void initialise();
}
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_SUPPORTED

#ifdef ALPAKA_ACC_GPU_CUDA_SUPPORTED
namespace alpaka_cuda_async {
  void initialise();
}
#endif  // ALPAKA_ACC_GPU_CUDA_SUPPORTED

// initialise all supported devices
void initialise() {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SUPPORTED
  alpaka_serial_sync::initialise();
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_SUPPORTED
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_SUPPORTED
  alpaka_tbb_async::initialise();
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_SUPPORTED
#ifdef ALPAKA_ACC_GPU_CUDA_SUPPORTED
  alpaka_cuda_async::initialise();
#endif  // ALPAKA_ACC_GPU_CUDA_SUPPORTED
}
