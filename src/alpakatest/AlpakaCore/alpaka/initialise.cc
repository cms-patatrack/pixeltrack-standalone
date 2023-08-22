#include <iostream>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaDevices.h"
#include "AlpakaCore/initialise.h"
#include "Framework/demangle.h"

namespace cms::alpakatools {

  template <typename TPlatform>
  void initialise(bool verbose) {
    constexpr const char* suffix[] = {"devices.", "device:", "devices:"};

    if (not platform<TPlatform>.has_value()) {
      // global objects in the cms::alpakatools namespace
      platform<TPlatform> = TPlatform{};
      devices<TPlatform> = alpaka::getDevs(*platform<TPlatform>);

      auto size = devices<TPlatform>.size();
      std::cout << "Found " << size << " " << suffix[size < 2 ? size : 2] << std::endl;
      for (auto const& device : devices<TPlatform>) {
        std::cout << "  - " << alpaka::getName(device) << std::endl;
      }
      if (verbose) {
        std::cout << edm::demangle<TPlatform> << " platform succesfully initialised." << std::endl;
      }
    } else {
      if (verbose) {
        std::cout << edm::demangle<TPlatform> << " platform already initialised." << std::endl;
      }
    }
    std::cout << std::endl;
  }

  // explicit template instantiation definition
  template void initialise<ALPAKA_ACCELERATOR_NAMESPACE::Platform>(bool);

}  // namespace cms::alpakatools
