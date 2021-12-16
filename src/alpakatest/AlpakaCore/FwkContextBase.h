#ifndef HeterogeneousCore_AlpakaCore_FwkContextBase_h
#define HeterogeneousCore_AlpakaCore_FwkContextBase_h

#include <memory>

// Only for StreamID
#include "Framework/Event.h"

#include "AlpakaCore/alpakaConfigFwd.h"
#include "AlpakaCore/Context.h"
#include "AlpakaCore/ProductMetadata.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE::impl {
  /**
   * This class is a base class for other Context classes for interacting with the framework
   */
  class FwkContextBase {
  public:
    FwkContextBase(FwkContextBase const&) = delete;
    FwkContextBase& operator=(FwkContextBase const&) = delete;
    FwkContextBase(FwkContextBase&&) = delete;
    FwkContextBase& operator=(FwkContextBase&&) = delete;

    Context makeContext() {
      return Context(metadata_->queue_.get(), &queueUsed_);
    }

    std::shared_ptr<ProductMetadata> const& metadataPtr() {
      queueUsed_ = true;
      return metadata_;
    }

  protected:
    // The constructors set the current device, but the device
    // is not set back to the previous value at the destructor. This
    // should be sufficient (and tiny bit faster) as all CUDA API
    // functions relying on the current device should be called from
    // the scope where this context is. The current device doesn't
    // really matter between modules (or across TBB tasks).
    explicit FwkContextBase(edm::StreamID streamID);

    explicit FwkContextBase(std::shared_ptr<ProductMetadata> metadata) : metadata_(std::move(metadata)) {}

#ifdef TODO
    explicit FwkContextBase(int device);

    // meant only for testing
    explicit FwkContextBase(int device, SharedStreamPtr stream);
#endif

    bool queueUsed() const {
      return queueUsed_;
    }

    void setQueue(std::shared_ptr<Queue> q);

    Queue& queue() {
      queueUsed_ = true;
      return *metadata_->queue_;
    }

    std::shared_ptr<ProductMetadata> releaseMetadataPtr() {
      return std::move(metadata_);
    }

  private:
    std::shared_ptr<ProductMetadata> metadata_;
    bool queueUsed_ = false;
  };
}

#endif
