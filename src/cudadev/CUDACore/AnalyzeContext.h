#ifndef HeterogeneousCore_CUDACore_AnalyzeContext_h
#define HeterogeneousCore_CUDACore_AnalyzeContext_h

#include "CUDACore/EDGetterContextBase.h"

namespace cms::cuda {
  /**
   * The aim of this class is to do necessary per-event "initialization" in analyze()
   * - setting the current device
   * - synchronizing between CUDA streams if necessary
   * and enforce that those get done in a proper way in RAII fashion.
   */
  class AnalyzeContext : public impl::EDGetterContextBase {
  public:
    /// Constructor to (possibly) re-use a CUDA stream
    explicit AnalyzeContext(edm::StreamID streamID) : EDGetterContextBase(streamID) {}
  };
}

#endif
