#ifndef HeterogeneousCore_CUDACore_TaskContext_h
#define HeterogeneousCore_CUDACore_TaskContext_h

#include "CUDACore/FwkContextBase.h"
#include "Framework/WaitingTaskWithArenaHolder.h"

namespace cms::cuda {
  namespace impl {
    class FwkContextHolderHelper {
    public:
      FwkContextHolderHelper(edm::WaitingTaskWithArenaHolder waitingTaskHolder, int device)
        : waitingTaskHolder_{std::move(waitingTaskHolder)}, device_{device} {}

      template <typename F>
      void pushNextTask(F&& f);

      void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
        waitingTaskHolder_ = std::move(waitingTaskHolder);
      }

      void enqueueCallback(cudaStream_t stream);

    private:
      edm::WaitingTaskWithArenaHolder waitingTaskHolder_;
      int device_;
    };
  }

  /**
   * The aim of this class is to do necessary per-task "initialization" tasks created in ExternalWork acquire():
   * - setting the current device
   * - calling edm::WaitingTaskWithArenaHolder::doneWaiting() when necessary
   */
  class TaskContext : public impl::FwkContextBase {
  public:
    /// Constructor to re-use the CUDA stream of acquire() (ExternalWork module)
    explicit TaskContext(int device, edm::WaitingTaskWithArenaHolder waitingTaskHolder)
        : FwkContextBase(device), holderHelper_{std::move(waitingTaskHolder), device} {}

    ~TaskContext() = default;

    template <typename F>
    void pushNextTask(F&& f) {
      holderHelper_.pushNextTask(std::forward<F>(f));
    }

    void replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      holderHelper_.replaceWaitingTaskHolder(std::move(waitingTaskHolder));
    }

    // Internal API
    void commit();

  private:
    impl::FwkContextHolderHelper holderHelper_;
  };

  namespace impl {
    template <typename F>
    void FwkContextHolderHelper::pushNextTask(F&& f) {
      replaceWaitingTaskHolder(edm::WaitingTaskWithArenaHolder{edm::make_waiting_task_with_holder(
          tbb::task::allocate_root(),
          std::move(waitingTaskHolder_),
          [device = device_, func = std::forward<F>(f)](edm::WaitingTaskWithArenaHolder h) {
            func(TaskContext{device, std::move(h)});
          })});
    }
  }  // namespace impl
}

#endif
