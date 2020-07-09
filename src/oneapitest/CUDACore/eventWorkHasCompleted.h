#ifndef HeterogeneousCore_CUDAUtilities_eventWorkHasCompleted_h
#define HeterogeneousCore_CUDAUtilities_eventWorkHasCompleted_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CUDACore/cudaCheck.h"

namespace cms {
  namespace cuda {
    /**
   * Returns true if the work captured by the event (=queued to the
   * CUDA stream at the point of cudaEventRecord()) has completed.
   *
   * Returns false if any captured work is incomplete.
   *
   * In case of errors, throws an exception.
   */
    inline bool eventWorkHasCompleted(sycl::event event) try {
      const auto ret = (int)event.get_info<sycl::info::event::command_execution_status>();
      if (ret == 0) {
        return true;
      } else
      // leave error case handling to cudaCheck
      cudaCheck(ret);
      return false;  // to keep compiler happy
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
  }  // namespace cuda
}  // namespace cms

#endif
