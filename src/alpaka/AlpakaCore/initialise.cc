#include "AlpakaCore/alpakaConfigAcc.h"
#include "AlpakaCore/alpakaDevAcc.h"
#include "AlpakaCore/backend.h"
#include "AlpakaCore/initialise.h"

static constexpr const char* suffix(size_t size) {
  constexpr const char* devices[] = { "devices.", "device:", "devices:" };
  return devices[size < 2 ? size : 2];
}

namespace cms::alpakatools {

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template <>
  void initialise<Backend::SERIAL>() {
    using Platform = alpaka_serial_sync::Platform;
    if (devices<Platform>.empty()) {
      devices<Platform> = enumerate<Platform>();
      auto size = devices<Platform>.size();
      std::cout << "CPU platform succesfully initialised" << std::endl;
      std::cout << "Found " << size << " " << suffix(size) << std::endl;
      for (auto const& device: devices<Platform>) {
        std::cout << "  - " << alpaka::getName(device) << std::endl;
      }
    } else {
      std::cout << "CPU platform already initialised" << std::endl;
    }
  }
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  template <>
  void initialise<Backend::TBB>() {
    using Platform = alpaka_tbb_async::Platform;
    if (devices<Platform>.empty()) {
      devices<Platform> = enumerate<Platform>();
      auto size = devices<Platform>.size();
      std::cout << "CPU platform succesfully initialised" << std::endl;
      std::cout << "Found " << size << " " << suffix(size) << std::endl;
      for (auto const& device: devices<Platform>) {
        std::cout << "  - " << alpaka::getName(device) << std::endl;
      }
    } else {
      std::cout << "CPU platform already initialised" << std::endl;
    }
  }
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  template <>
  void initialise<Backend::CUDA>() {
    using Platform = alpaka_cuda_async::Platform;
    if (devices<Platform>.empty()) {
      devices<Platform> = enumerate<Platform>();
      auto size = devices<Platform>.size();
      std::cout << "CUDA platform succesfully initialised" << std::endl;
      std::cout << "Found " << size << " " << suffix(size) << std::endl;
      for (auto const& device: devices<Platform>) {
        std::cout << "  - " << alpaka::getName(device) << std::endl;
      }
    } else {
      std::cout << "CUDA platform already initialised" << std::endl;
    }
  }
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

}  // namespace cms::alpakatools
