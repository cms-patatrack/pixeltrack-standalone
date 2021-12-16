#ifndef HeterogeneousCore_AlpakaCore_Context_h
#define HeterogeneousCore_AlpakaCore_Context_h

#include "AlpakaCore/alpakaConfigFwd.h"

#include <memory>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * Lightweight, non-owning helper to deliver the Queue to the device kernels
   */
  class Context {
  public:
    Context(Queue* queue, bool* queueUsed) : queue_(queue), queueUsed_(queueUsed) {}

    // prevent copying and moving
    Context(Context const&) = delete;
    Context& operator=(Context const&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    Queue& queue() {
      *queueUsed_ = true;
      return *queue_;
    }

  private:
    Queue* queue_;
    bool* queueUsed_;
  };
}

#endif
