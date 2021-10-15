#ifndef HeterogeneousCore_AlpakaUtilities_eventWorkHasCompleted_h
#define HeterogeneousCore_AlpakaUtilities_eventWorkHasCompleted_h

#include "AlpakaCore/alpakaConfig.h"

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  /**
     * Returns true if the work captured by the event (=queued to the
     * CUDA stream at the point of cudaEventRecord()) has completed.
     *
     * Returns false if any captured work is incomplete.
     *
     * In case of errors, throws an exception.
   */

  inline bool eventWorkHasCompleted(alpaka::Event<::ALPAKA_ACCELERATOR_NAMESPACE::Queue> event) {
    return alpaka::isComplete(event);
  }

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaUtilities_eventWorkHasCompleted_h
