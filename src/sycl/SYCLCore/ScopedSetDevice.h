//TODO

#ifndef HeterogeneousCore_SYCLUtilities_ScopedSetDevice_h
#define HeterogeneousCore_SYCLUtilities_ScopedSetDevice_h

#include <sycl/sycl.hpp>
#include <vector>

namespace cms {
  namespace sycltools {
    class ScopedSetDevice {
    public:
      // Store the original device, without setting a new one
      ScopedSetDevice(sycl::queue stream) {
        // Store the original device
        originalDevice_ = stream.get_device();
        stream_ = stream;
      }

      // Store the original device, and set a new current device
      explicit ScopedSetDevice(sycl::device device, sycl::queue stream) {
        originalDevice_ = stream.get_device();
        // Change the current device
        stream = sycl::queue(device);
      }

      // Restore the original device
      ~ScopedSetDevice() { stream_ = sycl::queue(originalDevice_); }

      // Set a new current device, without changing the original device
      // that will be restored when this object is destroyed
      //void set(int device) {
      // Change the current device
      //std::vector<sycl::device> device_list = sycl::device::get_devices(sycl::info::device_type::all);
      //int dev_idx = distance(device_list.begin(), find(device_list.begin(), device_list.end(), device));
      //cudaCheck(cudaSetDevice(device));
      //}

    private:
      sycl::device originalDevice_;
      sycl::queue stream_;
    };
  }  // namespace sycltools
}  // namespace cms

#endif
