#ifndef AlpakaCore_HostOnlyTask_h
#define AlpakaCore_HostOnlyTask_h

#include <functional>
#include <memory>

#include <alpaka/alpaka.hpp>
#include <alpaka/alpakaExtra.hpp>

namespace alpaka {

  class HostOnlyTask {
  public:
    HostOnlyTask(std::function<void()> task) : task_(std::move(task)) {}

    void operator()() const { task_(); }

  private:
    std::function<void()> task_;
  };

  namespace traits {

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    //! The CUDA async queue enqueue trait specialization for "safe tasks"
    template <>
    struct Enqueue<QueueCudaRtNonBlocking, HostOnlyTask> {
      static void CUDART_CB callback(cudaStream_t /*queue*/, cudaError_t /*status*/, void* arg) {
        //ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(status);
        std::unique_ptr<HostOnlyTask> pTask(static_cast<HostOnlyTask*>(arg));
        (*pTask)();
      }

      ALPAKA_FN_HOST static auto enqueue(QueueCudaRtNonBlocking& queue, HostOnlyTask task) -> void {
        auto pTask = std::make_unique<HostOnlyTask>(std::move(task));
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            cudaStreamAddCallback(alpaka::getNativeHandle(queue), callback, static_cast<void*>(pTask.release()), 0u));
      }
    };
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    //! The HIP async queue enqueue trait specialization for "safe tasks"
    template <>
    struct Enqueue<QueueHipRtNonBlocking, HostOnlyTask> {
      static void HIPRT_CB callback(hipStream_t /*queue*/, hipError_t /*status*/, void* arg) {
        //ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(status);
        std::unique_ptr<HostOnlyTask> pTask(static_cast<HostOnlyTask*>(arg));
        (*pTask)();
      }

      ALPAKA_FN_HOST static auto enqueue(QueueHipRtNonBlocking& queue, HostOnlyTask task) -> void {
        auto pTask = std::make_unique<HostOnlyTask>(std::move(task));
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            hipStreamAddCallback(alpaka::getNativeHandle(queue), callback, static_cast<void*>(pTask.release()), 0u));
      }
    };
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

  }  // namespace traits

}  // namespace alpaka

#endif  // AlpakaCore_HostOnlyTask_h
