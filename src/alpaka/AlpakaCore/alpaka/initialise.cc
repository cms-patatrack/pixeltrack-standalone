#include <iostream>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/config.h"
#include "AlpakaCore/alpaka/devices.h"
#include "AlpakaCore/initialise.h"
#include "Framework/demangle.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void initialise(bool verbose) {
    constexpr const char* suffix[] = {"devices.", "device:", "devices:"};
    static bool done = false;

    if (not done) {
      auto size = cms::alpakatools::devices<Platform>().size();
      std::cout << "Found " << size << " " << suffix[size < 2 ? size : 2] << std::endl;
      for (auto const& device : cms::alpakatools::devices<Platform>()) {
        std::cout << "  - " << alpaka::getName(device) << std::endl;
      }
      if (verbose) {
        std::cout << edm::demangle<Platform> << " platform succesfully initialised." << std::endl;
      }
      std::cout << std::endl;
      done = true;
    } else {
      if (verbose) {
        std::cout << edm::demangle<Platform> << " platform already initialised." << std::endl;
        std::cout << std::endl;
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
