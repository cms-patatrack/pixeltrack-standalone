#ifndef EmptyWaitingTask_h
#define EmptyWaitingTask_h

// from FWCore/Concurrency/interface/WaitingTaskList.h
#include "Framework/WaitingTask.h"

namespace edm {
  class EmptyWaitingTask : public WaitingTask {
  public:
    EmptyWaitingTask() = default;

    tbb::task* execute() override { return nullptr; }
  };

  namespace waitingtask {
    struct TaskDestroyer {
      void operator()(tbb::task* iTask) const { tbb::task::destroy(*iTask); }
    };
  }  // namespace waitingtask
  ///Create an EmptyWaitingTask which will properly be destroyed
  inline std::unique_ptr<edm::EmptyWaitingTask, waitingtask::TaskDestroyer> make_empty_waiting_task() {
    return std::unique_ptr<edm::EmptyWaitingTask, waitingtask::TaskDestroyer>(new (tbb::task::allocate_root())
                                                                                  edm::EmptyWaitingTask{});
  }
}  // namespace edm

#endif
