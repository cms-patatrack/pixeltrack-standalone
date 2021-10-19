#include <stdexcept>

#include "AlpakaCore/ContextState.h"

namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE {

  void ContextState::throwIfStream() const {
    if (stream_) {
      throw std::runtime_error("Trying to set ContextState, but it already had a valid state");
    }
  }

  void ContextState::throwIfNoStream() const {
    if (not stream_) {
      throw std::runtime_error("Trying to get ContextState, but it did not have a valid state");
    }
  }

}  // namespace cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE
