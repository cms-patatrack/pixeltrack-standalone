#ifndef HeterogeneousCore_CUDAUtilities_eventWorkHasCompleted_h
#define HeterogeneousCore_CUDAUtilities_eventWorkHasCompleted_h

#include "CUDACore/cudaCheck.h"

#include <hip/hip_runtime.h>

namespace cms {
  namespace cuda {
    /**
   * Returns true if the work captured by the event (=queued to the
   * CUDA stream at the point of hipEventRecord()) has completed.
   *
   * Returns false if any captured work is incomplete.
   *
   * In case of errors, throws an exception.
   */
    inline bool eventWorkHasCompleted(hipEvent_t event) {
      const auto ret = hipEventQuery(event);
      if (ret == hipSuccess) {
        return true;
      } else if (ret == hipErrorNotReady) {
        return false;
      }
      // leave error case handling to cudaCheck
      cudaCheck(ret);
      return false;  // to keep compiler happy
    }
  }  // namespace cuda
}  // namespace cms

#endif
