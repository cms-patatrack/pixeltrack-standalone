#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>

#include "CUDACore/requireDevices.h"

namespace cms::cudatest {
  bool testDevices() {
    int devices = 0;
    auto status = hipGetDeviceCount(&devices);
    if (status != hipSuccess) {
      std::cerr << "Failed to initialise the CUDA runtime, the test will be skipped."
                << "\n";
      return false;
    }
    if (devices == 0) {
      std::cerr << "No CUDA devices available, the test will be skipped."
                << "\n";
      return false;
    }
    return true;
  }

  void requireDevices() {
    if (not testDevices()) {
      exit(EXIT_SUCCESS);
    }
  }
}  // namespace cms::cudatest
