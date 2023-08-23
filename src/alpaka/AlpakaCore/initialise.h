#ifndef AlpakaCore_initialise_h
#define AlpakaCore_initialise_h

#include "AlpakaCore/config.h"

namespace cms::alpakatools {

  // note: this function is not thread-safe
  template <typename TPlatform>
  void initialise(bool verbose = false);

  // explicit template instantiation declaration
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT
  extern template void initialise<alpaka_serial_sync::Platform>(bool);
#endif
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_PRESENT
  extern template void initialise<alpaka_tbb_async::Platform>(bool);
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_PRESENT
  extern template void initialise<alpaka_cuda_async::Platform>(bool);
#endif
#ifdef ALPAKA_ACC_GPU_HIP_PRESENT
  extern template void initialise<alpaka_rocm_async::Platform>(bool);
#endif

}  // namespace cms::alpakatools

#endif  // AlpakaCore_initialise_h
