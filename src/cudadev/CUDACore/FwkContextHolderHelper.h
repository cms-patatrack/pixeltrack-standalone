#ifndef HeterogeneousCore_CUDACore_FwkContextHolderHelper_h
#define HeterogeneousCore_CUDACore_FwkContextHolderHelper_h

namespace cms::cuda::impl {
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

#endif
