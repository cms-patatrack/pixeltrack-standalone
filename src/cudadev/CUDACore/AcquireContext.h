#ifndef HeterogeneousCore_CUDACore_AcquireContext_h
#define HeterogeneousCore_CUDACore_AcquireContext_h

#include "CUDACore/EDGetterContextBase.h"
#include "CUDACore/TaskContext.h"

namespace cms::cuda {
  /**
   * The aim of this class is to do necessary per-event "initialization" in ExternalWork acquire():
   * - setting the current device
   * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
   * - synchronizing between CUDA streams if necessary
   * Users should not, however, construct it explicitly.
   */
  class AcquireContext : public impl::EDGetterContextBase {
  public:
    explicit AcquireContext(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : EDGetterContextBase(streamID), holderHelper_{std::move(waitingTaskHolder), device()} {}
    ~AcquireContext() = default;

    template <typename F>
    void pushNextTask(F&& f) {
      holderHelper_.pushNextTask(std::forward<F>(f));
    }

    void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
    }

    // internal API
    void commit();

  private:
    impl::FwkContextHolderHelper holderHelper_;
  };

  template <typename F>
  void runAcquire(edm::StreamID streamID, edm::WaitingTaskWithArenaHolder holder, F func) {
    AcquireContext context(streamID, std::move(holder));
    func(context);
    context.commit();
  }
}

#endif
