#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

#include "chooseDevice.h"

namespace cms::sycltools {

  static std::vector<sycl::device> discoverDevices() {
    std::vector<sycl::device> temp;
    std::vector<sycl::device> cpus = sycl::device::get_devices(sycl::info::device_type::cpu);
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (auto it = cpus.begin(); it != cpus.end(); it++) {
      if (it + 1 == cpus.end()) {
        break;
      }
      if ((*it).get_info<sycl::info::device::name>() == (*(it + 1)).get_info<sycl::info::device::name>() and
          (*it).get_backend() == (*(it + 1)).get_backend() and
          (*(it + 1)).get_info<sycl::info::device::driver_version>() <
              (*it).get_info<sycl::info::device::driver_version>()) {
        cpus.erase(it + 1);
      }
    }
    temp.insert(temp.end(), cpus.begin(), cpus.end());

    for (auto it = gpus.begin(); it != gpus.end(); it++) {
      if (it + 1 == gpus.end()) {
        break;
      }
      if ((*it).get_info<sycl::info::device::name>() == (*(it + 1)).get_info<sycl::info::device::name>() and
          (*it).get_backend() == (*(it + 1)).get_backend() and
          (*(it + 1)).get_info<sycl::info::device::driver_version>() <
              (*it).get_info<sycl::info::device::driver_version>()) {
        gpus.erase(it + 1);
      }
    }
    temp.insert(temp.end(), gpus.begin(), gpus.end());
    return temp;
  }

  std::vector<sycl::device> const& enumerateDevices(bool verbose) {
    static const std::vector<sycl::device> devices = discoverDevices();

    if (verbose) {
      std::cerr << "Found " << devices.size() << " SYCL devices:" << std::endl;
      for (auto const& device : devices)
        std::cerr << "  - " << device.get_backend() << ' ' << device.get_info<sycl::info::device::name>() << " ["
                  << device.get_info<sycl::info::device::driver_version>() << "]" << std::endl;
      std::cerr << std::endl;
    }
    return devices;
  }

  static std::vector<sycl::platform> discoverPlatforms() {
    std::vector<sycl::platform> temp;
    auto const& devices = enumerateDevices();

    for (auto dev : devices) {
      if (std::find(temp.begin(), temp.end(), dev.get_platform()) == temp.end()) {
        temp.emplace_back(dev.get_platform());
      }
    }

    return temp;
  }

  std::vector<sycl::platform> const& enumeratePlatforms(bool verbose) {
    static const std::vector<sycl::platform> platforms = discoverPlatforms();

    if (verbose) {
      std::cerr << "Found " << platforms.size() << " SYCL Platforms:" << std::endl;
      for (auto const& plt : platforms)
        std::cerr << "  - " << plt.get_info<sycl::info::platform::name>() << std::endl;
    }
    return platforms;
  }

  sycl::device chooseDevice(edm::StreamID id, bool verbose) {
    auto const& devices = enumerateDevices();
    auto const& device = devices[id % devices.size()];
    if (verbose) {
      std::cerr << "EDM stream " << id << " offload to " << device.get_info<sycl::info::device::name>()
                << " on backend " << device.get_backend() << std::endl;
    }
    return device;
  }
}  // namespace cms::sycltools
