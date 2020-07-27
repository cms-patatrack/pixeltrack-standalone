#ifndef HeterogeneousCore_SYCLUtilities_eventWorkHasCompleted_h
#define HeterogeneousCore_SYCLUtilities_eventWorkHasCompleted_h

#include <CL/sycl.hpp>

namespace cms {
  namespace sycl {
    /**
     * Returns true if the work captured by the event (queued to the
     * SYCL stream) has completed.
     *
     * Returns false if any captured work is incomplete.
     *
     * In case of errors, throws an exception.
     */

    inline bool eventWorkHasCompleted(::sycl::event event) {
      return (event.get_info<::sycl::info::event::command_execution_status>() == ::sycl::info::event_command_status::complete);
    }
  }  // namespace sycl
}  // namespace cms

#endif
