#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CUDACore/StreamCache.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/currentDevice.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"

namespace cms::cuda {
  void StreamCache::Deleter::operator()(sycl::queue *stream) const {
    if (device_ != -1) {
      ScopedSetDevice deviceGuard{device_};
      /*
      DPCT1003:41: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      cudaCheck((dpct::get_current_device().destroy_queue(stream), 0));
    }
  }

  // StreamCache should be constructed by the first call to
  // getStreamCache() only if we have CUDA devices present
  StreamCache::StreamCache() : cache_(deviceCount()) {}

  SharedStreamPtr StreamCache::get() {
    const auto dev = currentDevice();
    return cache_[dev].makeOrGet([dev]() {
      try {
        sycl::queue *stream;
        /*
      DPCT1003:42: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
        /*
      DPCT1025:43: The SYCL queue is created ignoring the flag/priority options.
      */
        cudaCheck((stream = dpct::get_current_device().create_queue(), 0));
      return std::unique_ptr<BareStream, Deleter>(stream, Deleter{dev});
      }
      catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
        std::exit(1);
      }
    });
  }

  void StreamCache::clear() {
    // Reset the contents of the caches, but leave an
    // edm::ReusableObjectHolder alive for each device. This is needed
    // mostly for the unit tests, where the function-static
    // StreamCache lives through multiple tests (and go through
    // multiple shutdowns of the framework).
    cache_.clear();
    cache_.resize(deviceCount());
  }

  StreamCache& getStreamCache() {
    // the public interface is thread safe
    static StreamCache cache;
    return cache;
  }
}  // namespace cms::cuda
