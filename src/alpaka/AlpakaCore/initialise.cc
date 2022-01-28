#include "AlpakaCore/alpakaConfigAcc.h"
#include "AlpakaCore/alpakaDevAcc.h"
#include "AlpakaCore/backend.h"
#include "AlpakaCore/initialise.h"

namespace cms::alpakatools {

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template <>
  void initialise<Backend::SERIAL>() {
    using Platform = alpaka_serial_sync::Platform;
    if (devices<Platform>.empty()) {
      devices<Platform> = enumerate<Platform>();
      std::cout << "platform initialised" << std::endl;
    } else {
      std::cout << "platform already initialised" << std::endl;
    }
  }
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  template <>
  void initialise<Backend::TBB>() {
    using Platform = alpaka_tbb_async::Platform;
    if (devices<Platform>.empty()) {
      devices<Platform> = enumerate<Platform>();
      std::cout << "platform initialised" << std::endl;
    } else {
      std::cout << "platform already initialised" << std::endl;
    }
  }
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  template <>
  void initialise<Backend::CUDA>() {
    using Platform = alpaka_cuda_async::Platform;
    if (devices<Platform>.empty()) {
      devices<Platform> = enumerate<Platform>();
      std::cout << "platform initialised" << std::endl;
    } else {
      std::cout << "platform already initialised" << std::endl;
    }
  }
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

}  // namespace cms::alpakatools
