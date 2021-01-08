#include "Framework/Worker.h"

namespace edm {
  void Worker::prefetchAsync(Event& event, EventSetup const& eventSetup, WaitingTask* iTask) {
    //std::cout << "prefetchAsync for " << this << " iTask " << iTask << std::endl;
    bool expected = false;
    if (prefetchRequested_.compare_exchange_strong(expected, true)) {
      //std::cout << "first prefetch call" << std::endl;
      //Need to be sure the ref count isn't set to 0 immediately
      iTask->increment_ref_count();
      for (Worker* dep : itemsToGet_) {
        //std::cout << "calling doWorkAsync for " << dep << " with " << iTask << std::endl;
        dep->doWorkAsync(event, eventSetup, iTask);
      }

      auto count = iTask->decrement_ref_count();
      //std::cout << "count " << count << std::endl;
      if (0 == count) {
        //std::cout << "spawning iTask for " << this << " task " << iTask << std::endl;
        //if everything finishes before we leave this routine, we need to launch the task
        tbb::task::spawn(*iTask);
      }
    }
  }
}  // namespace edm
