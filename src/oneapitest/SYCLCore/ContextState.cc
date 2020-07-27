#include "SYCLCore/ContextState.h"

#include <stdexcept>

namespace cms::sycl {
  void ContextState::throwIfStream() const {
    if (has_stream_) {
      throw std::runtime_error("Trying to set ContextState, but it already had a valid state");
    }
  }

  void ContextState::throwIfNoStream() const {
    if (not has_stream_) {
      throw std::runtime_error("Trying to get ContextState, but it did not have a valid state");
    }
  }
}  // namespace cms::sycl
