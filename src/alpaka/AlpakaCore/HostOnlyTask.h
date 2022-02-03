#ifndef AlpakaCore_HostOnlyTask_h
#define AlpakaCore_HostOnlyTask_h

#include <functional>
#include <memory>

#include <alpaka/alpaka.hpp>

namespace alpaka {

  class HostOnlyTask {
  public:
    HostOnlyTask(std::function<void()> task) : task_(std::move(task)) {}

    void operator()() const { task_(); }

  private:
    std::function<void()> task_;
  };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

  namespace traits {
    //! The CUDA/HIP RT async queue enqueue trait specialization for "safe tasks"
    template <>
    struct Enqueue<QueueUniformCudaHipRtNonBlocking, HostOnlyTask> {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
      static void CUDART_CB
#else
      static void HIPRT_CB
#endif
      uniformCudaHipRtCallback(ALPAKA_API_PREFIX(Stream_t) /*queue*/,
                               ALPAKA_API_PREFIX(Error_t) /*status*/,
                               void* arg) {
        //ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(status);
        std::unique_ptr<HostOnlyTask> pTask(static_cast<HostOnlyTask*>(arg));
        (*pTask)();
      }

      ALPAKA_FN_HOST static auto enqueue(QueueUniformCudaHipRtNonBlocking& queue, HostOnlyTask task) -> void {
        auto pTask = std::make_unique<HostOnlyTask>(std::move(task));
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            ALPAKA_API_PREFIX(StreamAddCallback)(queue.m_spQueueImpl->m_UniformCudaHipQueue,
                                                 uniformCudaHipRtCallback,
                                                 static_cast<void*>(pTask.release()),
                                                 0u));
      }
    };
  }  // namespace traits

#endif  // defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

}  // namespace alpaka

#endif  // AlpakaCore_HostOnlyTask_h
