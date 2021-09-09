#ifndef HeterogeneousCore_CUDACore_FwkContextBase_h
#define HeterogeneousCore_CUDACore_FwkContextBase_h

#include "CUDACore/ProductBase.h"
#include "CUDACore/SharedStreamPtr.h"
#include "Framework/Event.h"

namespace cms::cuda::impl {
  /**
   * This class is a base class for other Context classes for interacting with the framework
   */
  class FwkContextBase {
  public:
    FwkContextBase(FwkContextBase const&) = delete;
    FwkContextBase& operator=(FwkContextBase const&) = delete;
    FwkContextBase(FwkContextBase&&) = delete;
    FwkContextBase& operator=(FwkContextBase&&) = delete;
    
    int device() const { return currentDevice_; }

    cudaStream_t stream() {
      if (not isInitialized()) {
        initialize();
      }
      return stream_->streamPtr().get();
    }
    const SharedStreamPtr& streamPtr() {
      if (not isInitialized()) {
        initialize();
      }
      return stream_->streamPtr();
    }

  protected:
    // The constructors set the current device, but the device
    // is not set back to the previous value at the destructor. This
    // should be sufficient (and tiny bit faster) as all CUDA API
    // functions relying on the current device should be called from
    // the scope where this context is. The current device doesn't
    // really matter between modules (or across TBB tasks).
    explicit FwkContextBase(edm::StreamID streamID);

    explicit FwkContextBase(int device);

    // meant only for testing
    explicit FwkContextBase(int device, SharedStreamPtr stream);

    bool isInitialized() const { return bool(stream_); }

    void initialize();
    void initialize(const ProductBase& data);

    const std::shared_ptr<impl::StreamSharingHelper>& streamSharingHelper() {
      if (not isInitialized()) {
        initialize();
      }
      return stream_;
    }

  private:
    int currentDevice_ = -1;
    std::shared_ptr<impl::StreamSharingHelper> stream_;
  };
}

#endif
