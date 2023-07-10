#ifndef HeterogeneousCore_SYCLUtilities_eventWorkHasCompleted_h
#define HeterogeneousCore_SYCLUtilities_eventWorkHasCompleted_h

#include <sycl/sycl.hpp>

namespace cms {
  namespace sycltools {
    /**
   * Returns true if the work captured by the event (=queued to the
   * SYCL stream at the point of ext_oneapi_submit_barrier()) has completed.
   *
   * Returns false if any captured work is incomplete.
   */
    inline bool eventWorkHasCompleted(sycl::event const& event) {
      const auto ret = event.get_info<sycl::info::event::command_execution_status>();

      if (ret == sycl::info::event_command_status::complete) {
        return true;
      }
      return false;  // to keep compiler happy
    }
  }  // namespace sycltools
}  // namespace cms

#endif
