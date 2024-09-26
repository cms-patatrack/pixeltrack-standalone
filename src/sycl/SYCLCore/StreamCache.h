#ifndef SYCLCore_StreamCache_h
#define SYCLCore_StreamCache_h

#include <memory>
#include <vector>
#include <sstream>

#include "SYCLCore/chooseDevice.h"
#include "SYCLCore/getDeviceIndex.h"
#include "Framework/ReusableObjectHolder.h"

namespace cms::sycltools {

  namespace {

    void syclExceptionHandler(sycl::exception_list exceptions) {
      std::ostringstream msg;
      msg << "Caught asynchronous SYCL exception:";
      for (auto const& exc_ptr : exceptions) {
        try {
          std::rethrow_exception(exc_ptr);
        } catch (sycl::exception const& e) {
          msg << '\n' << e.what();
        }
        throw std::runtime_error(msg.str());
      }
    }

  }  // namespace

  class StreamCache {
  public:
    // StreamCache should be constructed by the first call to
    // getStreamCache() only if we have CUDA devices present
    StreamCache() : cache_(cms::sycltools::enumerateDevices().size()) {}

    // Gets a (cached) CUDA stream for the current device. The stream
    // will be returned to the cache by the shared_ptr destructor.
    // This function is thread safe
    std::shared_ptr<sycl::queue> get(sycl::device const& dev) {
      return cache_[cms::sycltools::getDeviceIndex(dev)].makeOrGet([dev]() {
        return std::make_unique<sycl::queue>(dev, syclExceptionHandler, sycl::property::queue::in_order());
      });
    }

  private:
    // not thread safe, intended to be called only from CUDAService destructor
    void clear() {
      // Reset the contents of the caches, but leave an
      // edm::ReusableObjectHolder alive for each device. This is needed
      // mostly for the unit tests, where the function-static
      // StreamCache lives through multiple tests (and go through
      // multiple shutdowns of the framework).
      cache_.clear();
      cache_.resize(cms::sycltools::enumerateDevices().size());
    }

    std::vector<edm::ReusableObjectHolder<sycl::queue>> cache_;
  };

  // Gets the global instance of a StreamCache
  // This function is thread safe
  inline StreamCache& getStreamCache() {
    // the public interface is thread safe
    static StreamCache cache;
    return cache;
  }

}  // namespace cms::sycltools

#endif  // SYCLCore_StreamCache_h
