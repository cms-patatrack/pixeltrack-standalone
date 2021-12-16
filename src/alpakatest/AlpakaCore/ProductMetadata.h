#ifndef HeterogeneousCore_AlpakaCore_ProductMetadata_h
#define HeterogeneousCore_AlpakaCore_ProductMetadata_h

#include <atomic>
#include <memory>

#include "AlpakaCore/alpakaConfigFwd.h"
#include "Framework/WaitingTaskWithArenaHolder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace impl {
    class FwkContextBase;
  }

  class ProductMetadata {
  public:
    ProductMetadata(std::shared_ptr<Queue> queue, std::shared_ptr<AlpakaEvent> event) : queue_(std::move(queue)), event_(std::move(event)) {}
    ~ProductMetadata();

    void synchronizeWith(Queue& queue) const;

    std::shared_ptr<Queue> tryReuseQueue() const {
      bool expected = true;
      if(mayReuseStream_.compare_exchange_strong(expected, false)) {
        // If the current thread is the one flipping the flag, it may
        // reuse the stream.
        return queue_;
      }
      return nullptr;
    }

    void recordEvent();

    void enqueueCallback(edm::WaitingTaskWithArenaHolder holder);

  private:
    friend class ::ALPAKA_ACCELERATOR_NAMESPACE::impl::FwkContextBase;

    void setQueue(std::shared_ptr<Queue> queue);

    std::shared_ptr<Queue> queue_;
    std::shared_ptr<AlpakaEvent> event_;
    // This flag tells whether the Queue may be reused by a
    // consumer or not. The goal is to have a "chain" of modules to
    // queue their work to the same stream.
    mutable std::atomic<bool> mayReuseStream_ = true;
  };
}

#endif
